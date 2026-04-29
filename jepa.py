"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):
    # The actual latent dynamics JEPA model plus planning cost function

    def __init__(
        self,
        encoder, # turn images into embeddings
        predictor, # predict future frames latent embeddings from history frame and action embeddings
        action_encoder, # turns action vectors into action embeddings
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and (optional) actions into embeddings. 
        info: dict with pixels and action keys
        """
        # encode(): takes info["pixels"] and info["action"] and encode it into info["emb"] and info["act_emb"]

        pixels = info['pixels'].float()
        b = pixels.size(0) # b = batch size (e.g. 4), the size of the element at index 0, which is the batch
        # flatten for encoding: (B, T, C, H, W) -> (B*T, C, H, W), e.g. (32, 3, 3, 224, 224) -> (96, 3, 224, 224)
        # Because: image encoder expects a plain batch of images, not a batch of sequences.
        pixels = rearrange(pixels, "b t ... -> (b t) ...") 
        output = self.encoder(pixels, interpolate_pos_encoding=True) # send flattened images to the encoder
        # [:, 0]: ":" means all rows (all images in this batch),
        # and 0 means the first column (the CLS token, which is the learned embedding for the whole image).
        pixels_emb = output.last_hidden_state[:, 0]
        emb = self.projector(pixels_emb) # project the pixels_emb dimension to the model dimension
        # rearrange the embedding from (B*T, d) back to the original shape (B, T, D), e.g. (32, 3, 192), then append the ["emb"] key and value to the info dict
        # This is because later the component wants embeddings grouped by sample and time again
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"]) # encode the action tokens into action embeddings, and append to the info dict

        return info # the dict with two more fields appended: "emb" and "act_emb"

    def predict(self, emb, act_emb):
        """Given past latent state and actions, predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)

        B: Batch size
        T: number of timesteps in the history window
        D: dimension of the latent state
        A_emb: dimension of the action embeddings
        """
        preds = self.predictor(emb, act_emb) # The next latent state depends on both: 1. where you currently are (emb) 2. what actions were taken (act_emb)
        # flatten batch and time together. e.g. (32, 3, 512) -> (96, 512), because pred_proj is typically a simple per-vector projection head,
        # it doesn't need sequence structure, flattening lets it process every timestep embedding independently
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d")) 
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0)) # reshape the flattened result back into sequence form, later code expects temporal sequence again
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """
        Encodes history frames in info[pixels] into embeddings info[emb]

        Rollout the model given an initial info dict and action sequence.
        pixels (contains history frames): (B (Batch size), S (number of action plan samples), H (number of observed history frames), C (channels), H (height), W (width))
        -  S: the number of action plan samples, pixels contains S is because each sampled action is paired with the same current observation history frames (copied 300 times) for planning in parallel.
        action_sequence (contains sampled action sequences): (B, S, T, action_dim)
         - S: the number of action plan samples
         - T: the number of planned action steps per action plan / planning horizon / time horizon

         Value:
         pixels:          (1, 300, 1, 3, 224, 224)
         action_sequence: (1, 300, 5, 10)
        """

        # check the input dict contains a "pixels" key, because rollout needs initial observed frames, otherwise planning can't start
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        # split the action sequence into two parts along dimension 2 (T: planning horizon):
        # - act_0:      the first H(1) action step
        # - act_future: the remaining future T(5) - H(1) = 4 action steps
        # resulting tensor: act_0.shape = (1, 300, 1, 10)
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        # append action sequences with only the first step onto the info dict
        info["action"] = act_0 # later can encode both pixels and actions
        n_steps = T - H # 5 - 1 = 4, meaning 4 future rollout steps

        # copy and encode initial info dict
        # loop through all items in info (however, in this case, only 'pixels' is used), keep all items at dimension 0 (B: batch), only keep the first item at dimension 0 (S: sampled actions), then replace the original value.
        # Resulting vector:
        # info["pixels"]: (1, 300, 1, 3, 224, 224)   v[:, 0] -> (1, 1, 3, 224, 224): the same history frame tensor shared for 300 action sequences, only needs to encode one frame, then the embedding is copied 300 times.

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init) # encode the history frame into embeddings (1(B), 1(H), D)
        # unsqueeze: inserting a dimension with fixed size 1 at position 1, because the code wants a separate dimension for candidate plans 'S', result: [1(B), 1(S), 1(H), D]
        # expand: increase the dimension size of S from 1 to 300 without copying the data by creating views pointing to the same memory location, result: [1(B), 300(S), 1(H), D], -1 means kept unchanged.
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        # detach the tensor from the computation graph, so the gradients don't flow back to the original tensors, and create a new tensor with the same data, but not tied to the original computation graph. 
        _init = {k: detach_clone(v) for k, v in _init.items()} 

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone() # merge b(1) and s(300) into one dimension so the code can treat each (batch item, action sequence sample) as an indepedent rollout, makes a real copy
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        """
        Autoregressive rollout:
        1. take the embeddings of currently observed history frame embeddings (emb) and action embeddings (act_emb)
        2. predict the next state embedding (pred_emb)
        3. concatenate the predicted embedding to the history window (emb)
        4. repeat the process for the next step (t+1)
        5. until the end of the action sequence (t+n_steps)
        6. return the final predicted embedding (emb)

        Tensor Shape (Before rollout):
        emb: (300, 1, D)
        act: (300, 1, 10)
        act_future: (300, 4, 10)
        HS = history_size = 3
        n_steps = 4
        """
        HS = history_size # The predictor only looks at the most recent 3 state embeddings and action embeddings.
        for t in range(n_steps): # this loop runs 4 times, each iteration predicts one more future latent state
            act_emb = self.action_encoder(act) # encode the current action into action embeddings -> (300, 1, A_emb)
            emb_trunc = emb[:, -HS:]      # (B, H, D): keep batch size (300), take the last 3 timesteps along the temporal dimension, if < 3, take all available ones 
            act_trunc = act_emb[:, -HS:]  # (B, H, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D): take the last emb and A_emb and predict the next emb
            emb = torch.cat([emb, pred_emb], dim=1)                # (BS, T+1, D): join along the time dimension (e.g. 1st iter: (300, 1, D) + (300, 1, D) -> (300, 2, D), 2nd iter: (300, 2, D) + (300, 1, D) -> (300, 3, D)) 

            next_act = act_future[:, t : t + 1, :]   # (BS, 1, action_dim): This select next future action token for 300 rollouts
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim): Append that selected future action tokens to the running action history

        # predict the last state: Because if you have T=5 action steps, you want the latent state after applying the last action too. So: 5 actions, 6 states along the predicted output
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]            # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]        # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions, downstream code expects batch item, sampled action plan, predicted state, dimension
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout # (1, 300, 6, D): 1 env, 300 sampled action plans, each plan has 6 predicted state embeddings, each of shape (D)

        return info

    def criterion(self, info_dict: dict):
        """Compute the MSE cost between the final predicted embeddings and goal embeddings. Output one scalar cost for each sampled action plan."""
        pred_emb = info_dict["predicted_emb"]  # (B, S, T-1, dim)
        goal_emb = info_dict["goal_emb"]       # (B, S, T, dim)

        # keep all preceding dimensions, take last timestep in the time dimension (T), keep all embedding features (dim).
        # Transformation: (1, 300, 6, D) -> (1, 300, 1, D) --expand_as-> (1, 300, 6, D) to broadcast the same goal embedding at every timestep position, even though later it only uses the last timestep cost.
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb) 

        # return last-step cost per action candidate
        cost = F.mse_loss(                  # each element-wise MSE loss is (pred - goal)^2
            pred_emb[..., -1:, :],          # for each candidate plan, take only the last timestep predicted embedding (1, 300, 1, D)
            goal_emb[..., -1:, :].detach(), # also take the last goal latent state (same shape as pred_emb), and detach() stop gradients from flowing into the goal embedding tensor
            reduction="none", 
            # At this point (before sum), the output also has shape (1, 300, 1, D), means for each batch item, for each candidate plan, for the last timestep, you get a vector of MSE values for each embedding coordinate. e.g. MSE cost: [0.25, 1, 25, 16, ..., D]
            # So each candidate action plan still has 'D' number of loss, not summup yet to a scalar cost value. 
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S): This sum over the timestep dim 2, and embedding dim 3, this gives exactly one scalar per action candidate. (1, 300)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """
        get_cost(): planner entry point for JEPA, the planner calls this function to output the cost for each of the given action candidates, provided the info dict with goal and initial state.

        So the planner sees JEPA as a black box: input: 300 action plans, output: 300 scalar costs 

        Tensor shape:
            action_candidates: (B, S, T, action_dim) = (1, 300, 5, 10)
            info_dict["goal_emb"]: (B, S, T, D)
        """

        assert "goal" in info_dict, "goal not in info_dict" # Sanity check: planning only makes sense if you have a goal frame

        device = next(self.parameters()).device # all tensors used by the model should be moved to the same divice as the model
        # loop through all keys in the info dict, and if the value is a tensor, move it to the same device as the model
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                tensor = info_dict[k]
                # MPS does not support float64 tensors; cast env/state tensors down.
                if device.type == "mps" and tensor.is_floating_point() and tensor.dtype == torch.float64:
                    tensor = tensor.float()
                info_dict[k] = tensor.to(device)
        
        # create a goal dict, keep all batch items, take index 0 from the action samples, because 300 candidate plans share the same goal frame
        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)} 
        goal["pixels"] = goal["goal"] # encoder expects 'pixels'

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k) # encoder expects normal names like state, the goal-side versions should be renamed to match

        goal.pop("action") # remove action from the goal dict because we don't need to encode action part in the goal dict
        goal = self.encode(goal) # now, goal["emb"] should contain only the goal latent embedding 

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates) # calls the rollout function to output the predicted frame embeddings for each sampled action plan given the current observed frame

        cost = self.criterion(info_dict) # compute the cost for each candidate action plan (300 candidate action sequences)
        
        return cost
