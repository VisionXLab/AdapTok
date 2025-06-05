import torch
import pulp
from tqdm import tqdm
from scipy.stats import truncnorm

class MaskSampler:
    def __init__(self,
                 mask_type: str,
                 min_toks: int,
                 max_toks: int,
                 total_toks: int,
                 min_first_toks: int=1,
                 tot_groups: int=None,
                 eval_mask_type: str="full_ones",
                 mean_toks: int=1,
                 std_toks: int=1,
                 ):
        self.mask_type = mask_type
        self.min_toks = min_toks
        self.max_toks = max_toks
        self.mean_toks = mean_toks
        self.std_toks = std_toks
        self.total_toks = total_toks
        self.tot_groups = tot_groups
        self.min_first_toks = min_first_toks
        self.eval_mask_type = eval_mask_type

    @torch._dynamo.disable
    def __call__(self, num_items=1, device="cuda", is_training=True, keep_toks=-1, diff_batch=False, num_latents=None):
        mask_type = self.mask_type if is_training else self.eval_mask_type

        if mask_type == "left_masking":
            masks = self._left_masking(num_items, device, is_training, keep_toks, diff_batch)
        elif mask_type == "random_masking":
            masks = self._random_masking(num_items, device, is_training, keep_toks, diff_batch)
        elif mask_type == "left_masking_fixed":
            masks = self._left_masking_fixed(num_items, device)
        elif mask_type == "left_masking_by_group":
            masks = self._left_masking_by_group(num_items, device, diff_batch)
        elif mask_type == "left_masking_by_group_normal":
            masks = self._left_masking_by_group_normal(num_items, device, diff_batch)
        elif mask_type == "left_masking_by_group_fixed":
            masks = self._left_masking_by_group_fixed(num_items, device)
        elif mask_type == "left_masking_by_group_adap":
            masks = self._left_masking_by_group_adap(num_latents, device)
        elif mask_type == "full_ones":
            masks = self._full_ones(num_items, device)
        elif mask_type == "left_masking_by_group_half_ones":
            masks = self._left_masking_by_group_half(num_items, device)
        elif mask_type == "none_masking":
            return None
        else:
            raise NotImplementedError
        return masks.bool()

    def _left_masking_by_group_adap(self, num_latents=1, device="cuda"):
        num_items, num_groups = num_latents.shape
        assert num_groups == self.tot_groups
        idx = torch.arange(self.total_toks, device=device).view(1, 1, -1)
        masks = (idx < num_latents.unsqueeze(-1)).int()
        masks = masks.flatten(1)

        return masks

    def _left_masking_by_group_fixed(self, num_items=1, device="cuda"):
        masks = torch.zeros((num_items, self.tot_groups, self.total_toks), dtype=torch.int64, device=device)
        masks[:, :, :self.max_toks] = 1
        masks = masks.flatten(1)

        return masks

    def _left_masking_by_group_half(self, num_items=1, device="cuda"):
        masks = torch.zeros((num_items, self.tot_groups, self.total_toks), dtype=torch.int64, device=device)
        masks[:, :, :self.total_toks // 2] = 1
        masks = masks.flatten(1)

        return masks

    def _left_masking_by_group(self, num_items=1, device="cuda", diff_batch=False):
        latent_masks = []
        for i in range(num_items):
            num_kepts = torch.randint(self.min_toks, self.max_toks + 1, (self.tot_groups,))
            masks = torch.zeros((self.tot_groups, self.total_toks), dtype=torch.int64, device=device)
            for i, num_kept in enumerate(num_kepts):
                masks[i, :num_kept] = 1
            if self.min_first_toks > 0:
                masks[0, :self.min_first_toks] = 1
            if not diff_batch:
                return masks.flatten().repeat(num_items, 1) 
            
            latent_masks.append(masks.reshape(1, -1))

        masks = torch.cat(latent_masks, dim=0)

        return masks

    def _left_masking_by_group_normal(self, num_items=1, device="cuda", diff_batch=False):
        latent_masks = []
        for i in range(num_items):
            samples = truncnorm.rvs((self.min_toks - self.mean_toks) / self.std_toks,
                                    (self.max_toks - self.mean_toks) / self.std_toks,
                                    loc=self.mean_toks, scale=self.std_toks, size=self.tot_groups)
            num_kepts = torch.tensor(samples).clamp(self.min_toks, self.max_toks).round().to(dtype=torch.int)
            masks = torch.zeros((self.tot_groups, self.total_toks), dtype=torch.int64, device=device)
            for i, num_kept in enumerate(num_kepts):
                masks[i, :num_kept] = 1
            if self.min_first_toks > 0:
                masks[0, :self.min_first_toks] = 1
            if not diff_batch:
                return masks.flatten().repeat(num_items, 1) 
            
            latent_masks.append(masks.reshape(1, -1))

        masks = torch.cat(latent_masks, dim=0)

        return masks

    def _full_ones(self, num_items=1, device="cuda"):
        masks = torch.ones((num_items, self.total_toks * self.tot_groups), dtype=torch.int64, device=device)
        return masks

    def _left_masking_fixed(self, num_items=1, device="cuda"):
        masks = torch.zeros((num_items, self.tot_groups * self.total_toks), dtype=torch.int64, device=device)
        masks[:, :self.max_toks] = 1

        return masks

    def _left_masking(self, num_items=1, device="cuda", is_training=True, keep_toks=-1):
        if is_training:
            num_kepts = torch.randint(self.min_toks, self.max_toks + 1, (1,))
            masks = torch.zeros((num_items, self.total_toks), dtype=torch.int64, device=device)
            masks[:, :num_kepts] = 1
        else:
            masks = torch.ones((num_items, self.total_toks), dtype=torch.int64, device=device)
            if isinstance(keep_toks, list):
                for i, keep_tok in enumerate(keep_toks):
                    masks[i, keep_tok:] = 0
            elif keep_toks > 0:
                masks[:, keep_toks:] = 0
        return masks

    def _random_masking(self, num_items=1, device="cuda", is_training=True, keep_toks=-1):
        masks = torch.zeros((num_items, self.total_toks), dtype=torch.int64, device=device)
        if is_training:
            num_kepts = torch.randint(self.min_toks, self.max_toks + 1, (1,))
        else:
            num_kepts = keep_toks if keep_toks > 0 else self.total_toks
        
        for i in range(num_items):
            kept_indices = torch.randperm(self.total_toks, device=device)[:num_kepts]
            masks[i, kept_indices] = 1
        
        return masks

def left_masking_by_group_adap(num_latents, toks_per_group, device="cuda"):
    idx = torch.arange(toks_per_group, device=device).view(1, 1, -1)
    masks = (idx < num_latents.unsqueeze(-1)).int()
    masks = masks.flatten(1)

    return masks.bool()

def insert_special_token_before_rearrange(z, latent_nums, toks_per_group, special_token=0):
    """
    Inserts one special token into each group at the latent_nums[i, j] position.
    Works with both 2D and 3D input z.
    
    Args:
        z: (B, N) or (B, N, D)
        latent_nums: (B, G)
        toks_per_group: int
        special_token: scalar (for 2D) or (D,) Tensor (for 3D)
    
    Returns:
        z_new: shape (B, N + G) (or (B, N + G, D))
        latent_nums_new: (B, G)  # unchanged
    """
    B, G = latent_nums.shape
    D = None
    z_extra_dim = False

    if z.dim() == 2:
        z = z.unsqueeze(-1)
        z_extra_dim = True

    B, N, D = z.shape
    total_tokens = toks_per_group * G
    assert N == total_tokens, f"Expected z.shape[1] == toks_per_group * G, got {N} vs {toks_per_group * G}"

    z = z.view(B, G, toks_per_group, D)  # (B, G, T, D)

    # Create full output with space for one extra token per group
    z_new = torch.zeros(B, G, toks_per_group + 1, D, dtype=z.dtype, device=z.device)
    
    # Copy everything before insert
    arange = torch.arange(toks_per_group, device=z.device).view(1, 1, -1)
    mask = arange < latent_nums.unsqueeze(-1)
    z_new[:, :, :toks_per_group][mask] = z[mask]

    # Insert special token at position latent_nums
    if not torch.is_tensor(special_token):
        special_token = torch.full((D,), special_token, dtype=z.dtype, device=z.device)
    elif special_token.ndim == 0:
        special_token = special_token.expand(D)
    
    batch_idx = torch.arange(B, device=z.device).view(-1, 1).expand(B, G)
    group_idx = torch.arange(G, device=z.device).view(1, -1).expand(B, G)
    token_idx = latent_nums  # (B, G)

    z_new[batch_idx, group_idx, token_idx] = special_token  # insert into correct place

    # Reshape back
    z_new = z_new.view(B, (toks_per_group + 1) * G, D)

    if z_extra_dim:
        z_new = z_new.squeeze(-1)

    return z_new


def remove_special_token_and_pad_blockwise(z, latent_nums, toks_per_group, special_token=100):
    """
    Remove special token from each group and pad blockwise.
    Args:
        z: (B, N)
        latent_nums: (B, G)
        toks_per_group: int
    Returns:
        z_out: (B, G * toks_per_group)
    """
    if z.dim() == 3:
        z = z.squeeze(-1)

    B, N = z.shape
    B1, G = latent_nums.shape
    assert B == B1

    # 1. Mask out special tokens
    mask = z != special_token
    z_nospecial = torch.zeros_like(z)
    z_nospecial[mask] = z[mask]  # (B, N)

    # 2. Sort each row to bring non-special tokens forward
    sorted_z = torch.zeros_like(z)
    sorted_mask = torch.zeros_like(z)

    sorted_indices = mask.float().argsort(dim=1, descending=True)  # (B, N), non-specials first
    sorted_z = torch.gather(z_nospecial, dim=1, index=sorted_indices)

    # 3. Compute where each (b, g) group should write
    out = torch.zeros((B, G, toks_per_group), dtype=z.dtype, device=z.device)

    r = torch.arange(toks_per_group, device=z.device).view(1, 1, toks_per_group)  # (1, 1, T)
    latent_nums_clamped = latent_nums.clamp(max=toks_per_group).unsqueeze(-1)     # (B, G, 1)
    mask_pos = r < latent_nums_clamped                                            # (B, G, T)

    # 4. Compute offsets in the sorted_z
    start_offset = torch.cat([
        torch.zeros((B, 1), device=z.device, dtype=latent_nums.dtype),
        latent_nums.cumsum(dim=1)[:, :-1]
    ], dim=1).unsqueeze(-1)  # (B, G, 1)

    src_pos = (start_offset + r) * mask_pos  # (B, G, T)
    src_pos_flat = src_pos.view(B, -1)       # (B, G*T)

    gathered = torch.gather(sorted_z, dim=1, index=src_pos_flat)  # (B, G*T)
    out = gathered.view(B, G * toks_per_group)
    return out


def sub_solve_ilp_min(total_losses, tgt_toks, step=1, device="cuda", batch_size=500):
    assigned_toks = []
    for idx, start_idx in tqdm(enumerate(range(0, len(total_losses), batch_size)), total=len(total_losses) // batch_size + 1, desc="Processing Batches"):
        end_idx = min(len(total_losses), start_idx + batch_size)
        assigned_toks.append(solve_ilp_min(total_losses[start_idx: end_idx], tgt_toks, step, device))
    
    assigned_toks = torch.cat(assigned_toks, dim=0)
    
    selected_losses = total_losses[torch.arange(total_losses.size(0)), assigned_toks // step - 1]
    equal_toks = (total_losses.mean(axis=0) < selected_losses.mean()).int().argmax()
    equal_toks = total_losses.shape[-1] - 1 if equal_toks == 0 else equal_toks.item()
    equal_toks = equal_toks * step + step
    print(tgt_toks, "-->", equal_toks)

    return assigned_toks

def solve_ilp_min(total_losses, tgt_toks, step=1, device="cuda"):
    """
    Solve an Integer Linear Programming (ILP) problem to minimize the sum of selected elements.

    Args:
        total_losses (np.ndarray or torch.Tensor): A 2D array of shape (N, K) containing the losses.
        tgt_toks (int): Target sum of column indices in the solution.
        step (int): Step size to scale the selected indices. Default is 1.
        device (str): Device to store the output tensor. Default is "cuda".

    Returns:
        torch.Tensor: A tensor containing the selected column indices scaled by `step`.
    """
    N, K = total_losses.shape
    # tgt_toks = total_losses.shape[0] * (tgt_toks // step - 1)
    tgt_toks = total_losses.shape[0] * tgt_toks // step - total_losses.shape[0]
    
    prob = pulp.LpProblem("Minimize_Selected_Sum", pulp.LpMinimize)

    # whether to select element (i, j)
    x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(K)] for i in range(N)]

    # Objective: Minimize the sum of selected elements
    prob += pulp.lpSum(total_losses[i, j] * x[i][j] for i in range(N) for j in range(K))

    # Constraint 1: Each row must select exactly one column
    for i in range(N):
        prob += pulp.lpSum(x[i][j] for j in range(K)) == 1

    # Constraint 2: The sum of selected column indices must equal tgt_toks
    prob += pulp.lpSum(j * x[i][j] for i in range(N) for j in range(K)) == tgt_toks

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    selected_indices = [-1] * N
    for i in range(N):
        for j in range(K):
            if pulp.value(x[i][j]) == 1:
                selected_indices[i] = j 
                break

    selected_indices = torch.tensor([(i + 1) * step for i in selected_indices], device=device)

    return selected_indices


def rearrange_drop_mask(tensor, mask):
    """
    Rearranges the elements in each row of the input tensor based on the given mask.
    
    For each row:
      - The elements corresponding to True in the mask are moved to the front,
        preserving their original order.
      - The remaining positions (where mask is False) are filled with 0.
    
    The function supports input tensor shapes of (b, n) and (b, n, d). The output tensor 
    has the same shape as the input tensor.

    Parameters:
      tensor (torch.Tensor): Input tensor of shape (b, n) or (b, n, d).
      mask (torch.Tensor): Boolean mask of shape (b, n), where True indicates the element 
                           should be moved to the front.
    
    Returns:
      torch.Tensor: Rearranged tensor with the same shape as the input.

    Example:
        tensor_2d = torch.tensor([[5, 1, 2, 3],
                                [4, 5, 6, 7]], dtype=torch.float)
        mask = torch.tensor([[True, True, False, True],
                            [True, False, False, True]], dtype=torch.bool)
        # Expected output:
        # [[5, 1, 3, 0],
        #  [4, 7, 0, 0]]
    """
    
    extra_dim = False
    if tensor.dim() == 2:
        extra_dim = True
        tensor = tensor.unsqueeze(-1)
    
    b, n, d = tensor.shape
    output = torch.zeros_like(tensor)
    cumsum = mask.cumsum(dim=1) - 1
    
    row_idx = torch.arange(b, device=tensor.device).unsqueeze(1).expand(b, n)
    
    new_col_idx = cumsum[mask]
    new_row_idx = row_idx[mask]
    
    output[new_row_idx, new_col_idx, :] = tensor[mask]
    
    if extra_dim:
        output = output.squeeze(-1)
    
    return output


def pad_latent_mask(latent_mask):
    """
    Args:
        latent_mask: A boolean tensor of shape [bs, 1024], where:
                     - True represents an existing token
                     - False represents an empty position
    
    Goal:
        Ensure that each row contains exactly `max_toks` number of 1s,
        where `max_toks` is the maximum token count across all rows.
        If a row has fewer than `max_toks` tokens, fill the missing ones
        from the leftmost zero positions.

    Returns:
        padmask: The modified mask with extra tokens added.
        pad_add: A mask indicating the newly added positions.
    """
    bs, seq_len = latent_mask.shape
    token_counts = latent_mask.sum(dim=1)
    max_toks = token_counts.max()
    deficits = max_toks - token_counts

    zeros = (~latent_mask).long()
    cumsum_zeros = zeros.cumsum(dim=1)
    pad_add = (zeros == 1) & (cumsum_zeros <= deficits.unsqueeze(1))
    padmask = latent_mask | pad_add

    return padmask, pad_add

@torch._dynamo.disable
def extract_padded_tokens(z, padded_latent_mask):
    """
    Args:
        z: Tensor of shape [bs, 1024, d], containing feature tokens.
        padded_latent_mask: Boolean mask of shape [bs, 1024], indicating selected tokens.

    Goal:
        Extract tokens based on `padded_latent_mask` and return a tensor of shape [bs, max_toks, d].

    Returns:
        selected_tokens: A tensor of shape [bs, max_toks, d] containing selected tokens.
    """
    bs, seq_len, d = z.shape
    assert padded_latent_mask.sum(dim=1).max() == padded_latent_mask.sum(dim=1).min()
    max_toks = padded_latent_mask.sum(dim=1).max().item()  # All rows have exactly max_toks 1s

    # Use boolean masking to select valid tokens
    selected_tokens = z[padded_latent_mask]  # Shape: [bs * max_toks, d]

    # Reshape into final batch-wise format
    selected_tokens = selected_tokens.view(bs, max_toks, d)  # [bs, max_toks, d]

    return selected_tokens


def decoder_attn_mask_with_latent_mask(decoder_attn_mask, latent_masks, bs, fill_diag=False):
    attn_mask_modified = decoder_attn_mask.unsqueeze(0).repeat(bs, 1, 1)
    if latent_masks is not None:
        _, seq_len = latent_masks.shape  # bs, seq_len
        batch_idx, latent_idx = torch.where(latent_masks == False)
        attn_mask_modified[batch_idx, latent_idx, :] = False
        attn_mask_modified[batch_idx, :, latent_idx] = False
        if fill_diag:
            indices = torch.arange(seq_len, device=attn_mask_modified.device)    # set the diagonal elements to True !!!
            attn_mask_modified[:, indices, indices] = True

    return attn_mask_modified


def generate_attention_mask(num_img_tok_each: int, num_latent_each: int, num_groups: int, attn_type="full_causal_type1", latent_first=False):
    """
    Generate an attention mask based on the given parameters.

    Args:
        num_img_tok_each (int): Number of tokens per group.
        num_latent_each (int): Number of latent tokens per group.
        num_groups (int): Number of groups.
        attn_type (str): Type of attention masks.

    Returns:
        torch.Tensor: The generated attention mask of shape (total_seq_len, total_seq_len),
                      where True indicates attention is allowed, and False means blocked.
    """

    if attn_type == "full_causal_type1":
        attn_mask = generate_attention_mask_full_causal_type1(num_img_tok_each, num_latent_each, num_groups)
    elif attn_type == "full_causal_type2":
        attn_mask = generate_attention_mask_full_causal_type2(num_img_tok_each, num_latent_each, num_groups)
    elif attn_type == "full_causal_type3":
        attn_mask = generate_attention_mask_full_causal_type3(num_img_tok_each, num_latent_each, num_groups)
    elif attn_type == "full_bi_attn":
        attn_mask = generate_attention_mask_full_bi_attn(num_img_tok_each, num_latent_each, num_groups)
    else:
        raise NotImplementedError
    
    if latent_first:
        attn_mask = reorder_attention_mask(attn_mask, num_img_tok_each, num_latent_each, num_groups)

    return attn_mask

def generate_attention_mask_full_causal_type1(num_img_tok_each: int, num_latent_each: int, num_groups: int):
    """
    Generate an attention mask based on the given parameters.

    Args:
        num_img_tok_each (int): Number of tokens per group.
        num_latent_each (int): Number of latent tokens per group.
        num_groups (int): Number of groups.

    Returns:
        torch.Tensor: The generated attention mask of shape (total_seq_len, total_seq_len),
                      where True indicates attention is allowed, and False means blocked.
    """
    total_img_tokens = num_groups * num_img_tok_each
    total_latents = num_groups * num_latent_each
    total_seq_len = total_img_tokens + total_latents

    # Initialize full attention mask
    mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool)

    # Fill in mask to allow attention within each group
    for i in range(num_groups):
        start_tok = i * num_img_tok_each
        start_latent = total_img_tokens + i * num_latent_each

        # Top Left: Tokens in the group can attend to themselves
        mask[start_tok:start_tok + num_img_tok_each, 0:start_tok + num_img_tok_each] = True
        # Top Right: Tokens can attend to their corresponding latents
        mask[start_tok:start_tok + num_img_tok_each, total_img_tokens:start_latent + num_latent_each] = True
        # Bottom Left: Latents can attend to their corresponding tokens
        mask[start_latent:start_latent + num_latent_each, 0:start_tok + num_img_tok_each] = True
        # Bottom Right: Latents in the group can attend to themselves
        mask[start_latent:start_latent + num_latent_each, total_img_tokens:start_latent + num_latent_each] = True
        
    return mask


def generate_attention_mask_full_causal_type2(num_img_tok_each: int, num_latent_each: int, num_groups: int):
    """
    Generate an attention mask based on the given parameters.

    Args:
        num_img_tok_each (int): Number of tokens per group.
        num_latent_each (int): Number of latent tokens per group.
        num_groups (int): Number of groups.

    Returns:
        torch.Tensor: The generated attention mask of shape (total_seq_len, total_seq_len),
                      where True indicates attention is allowed, and False means blocked.
    """
    total_img_tokens = num_groups * num_img_tok_each
    total_latents = num_groups * num_latent_each
    total_seq_len = total_img_tokens + total_latents

    # Initialize full attention mask
    mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool)

    # Fill in mask to allow attention within each group
    for i in range(num_groups):
        start_tok = i * num_img_tok_each
        start_latent = total_img_tokens + i * num_latent_each

        # Top Left: Image-to-Image
        mask[start_tok:start_tok + num_img_tok_each, start_tok:start_tok + num_img_tok_each] = True
        # Top Right: Image-to-Latent
        mask[start_tok:start_tok + num_img_tok_each, total_img_tokens:start_latent + num_latent_each] = True
        # Bottom Right: Latent-to-Latent
        mask[start_latent:start_latent + num_latent_each, total_img_tokens:start_latent + num_latent_each] = True
        
    return mask

def generate_attention_mask_full_causal_type3(num_img_tok_each: int, num_latent_each: int, num_groups: int):
    """
    Generate an attention mask based on the given parameters.

    Args:
        num_img_tok_each (int): Number of tokens per group.
        num_latent_each (int): Number of latent tokens per group.
        num_groups (int): Number of groups.

    Returns:
        torch.Tensor: The generated attention mask of shape (total_seq_len, total_seq_len),
                      where True indicates attention is allowed, and False means blocked.
    """
    total_img_tokens = num_groups * num_img_tok_each
    total_latents = num_groups * num_latent_each
    total_seq_len = total_img_tokens + total_latents

    # Initialize full attention mask
    mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool)

    # Fill in mask to allow attention within each group
    for i in range(num_groups):
        start_tok = i * num_img_tok_each
        start_latent = total_img_tokens + i * num_latent_each

        # Top Left: Tokens in the group can attend to themselves
        mask[start_tok:start_tok + num_img_tok_each, start_tok:start_tok + num_img_tok_each] = True
        # Top Right: Tokens can attend to their corresponding latents
        # Bottom Left: Latents can attend to their corresponding tokens
        mask[start_latent:start_latent + num_latent_each, start_tok:start_tok + num_img_tok_each] = True
        # Bottom Right: Latents in the group can attend to themselves
        mask[start_latent:start_latent + num_latent_each, total_img_tokens:start_latent + num_latent_each] = True
        
    return mask


def generate_attention_mask_full_bi_attn(num_img_tok_each: int, num_latent_each: int, num_groups: int):
    total_img_tokens = num_groups * num_img_tok_each
    total_latents = num_groups * num_latent_each
    total_seq_len = total_img_tokens + total_latents
    mask = torch.ones((total_seq_len, total_seq_len), dtype=torch.bool)

    return mask


def reorder_attention_mask(mask: torch.Tensor, num_img_tok_each: int, num_latent_each: int, num_groups: int):
    """
    Create a new index order where latents come first, then images.
    Reorder the attention mask to swap the position of image tokens and latent tokens.

    Args:
        mask (torch.Tensor): The original attention mask.
        num_img_tok_each (int): Number of image tokens per group.
        num_latent_each (int): Number of latent tokens per group.
        num_groups (int): Number of groups.

    Returns:
        torch.Tensor: The reordered attention mask.
    """
    total_img_tokens = num_groups * num_img_tok_each
    total_latents = num_groups * num_latent_each
    total_seq_len = total_img_tokens + total_latents

    # Create a new index order where latents come first, then images
    img_indices = torch.arange(total_img_tokens)
    latent_indices = torch.arange(total_latents) + total_img_tokens
    new_order = torch.cat([latent_indices, img_indices])

    # Reorder the mask by selecting rows and columns in the new order
    reordered_mask = mask[new_order][:, new_order]
    
    return reordered_mask



if __name__=="__main__":
    # Example usage 
    num_img_tok_each = 8
    num_latent_each = 8
    num_groups = 4
    bs = 2
    attn_type = "full_causal_type3"

    mask = generate_attention_mask(num_img_tok_each, num_latent_each, num_groups, attn_type, latent_first=False)
