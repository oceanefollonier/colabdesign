# from _typeshed import Self
import random, os
import jax
import jax.numpy as jnp
import numpy as np
import time
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.shared.utils import copy_dict, update_dict, Key, dict_to_str, to_float, softmax, categorical, to_list, copy_missing

####################################################
# AF_DESIGN - design functions
####################################################
#\
# \_af_design
# |\
# | \_restart
#  \
#   \_design
#    \_step
#     \_run
#      \_recycle
#       \_single
#
####################################################

def _create_crop_indices(crop_residues, target_len):
  """
  Create a numpy array of indices to keep based on crop_residues ranges.
  
  Args:
    crop_residues: List of [start, end] pairs defining ranges to keep (0-indexed, end exclusive)
    target_len: Total length of target chain
    
  Returns:
    numpy array of indices to keep
  """
  if crop_residues is None or len(crop_residues) == 0:
    return None
    
  indices_to_keep = []
  for start, end in crop_residues:
    # Handle negative indices (e.g., -1 means until the end)
    if end < 0:
      end = target_len
    indices_to_keep.extend(range(start, min(end, target_len)))
  
  return np.array(indices_to_keep, dtype=np.int32)

def _crop_target_residues(inputs, target_crop_indices, target_len, binder_len):
  """
  Crop target residues from inputs, keeping binder residues intact.
  
  Args:
    inputs: Input dictionary containing features
    target_crop_indices: Array of target indices to keep
    target_len: Original target length
    binder_len: Binder length
    
  Returns:
    Cropped inputs dictionary
  """
  # Create full indices: cropped target + all binder
  full_indices = np.concatenate([
    target_crop_indices,
    np.arange(target_len, target_len + binder_len)
  ])
  
  cropped_inputs = {}
  for key, val in inputs.items():
    if not hasattr(val, 'shape'):
      cropped_inputs[key] = val
      continue
      
    # Handle different dimensionalities
    if key in ['aatype', 'residue_index', 'seq_mask', 'asym_id', 'sym_id', 'entity_id']:
      # 1D arrays along residue dimension
      cropped_inputs[key] = val[full_indices]
    elif key in ['target_feat']:
      # 2D: (num_res, channels)
      cropped_inputs[key] = val[full_indices, :]
    elif key in ['msa_feat', 'msa_mask']:
      # 2D: (num_seq, num_res, ...)
      if val.ndim == 2:
        cropped_inputs[key] = val[:, full_indices]
      elif val.ndim == 3:
        cropped_inputs[key] = val[:, full_indices, :]
    elif key == 'batch':
      # Recursively crop batch dict
      cropped_inputs[key] = _crop_batch_dict(val, target_crop_indices, target_len, binder_len)
    elif key == 'prev':
      # Handle prev dict (prev_pos, prev_msa_first_row, prev_pair)
      cropped_inputs[key] = _crop_prev_dict(val, target_crop_indices, target_len, binder_len)
    else:
      # For other keys, try to infer cropping strategy
      if val.ndim == 1 and val.shape[0] == target_len + binder_len:
        cropped_inputs[key] = val[full_indices]
      elif val.ndim == 2:
        if val.shape[0] == target_len + binder_len:
          cropped_inputs[key] = val[full_indices, :]
        elif val.shape[1] == target_len + binder_len:
          cropped_inputs[key] = val[:, full_indices]
      else:
        # Keep as is if we can't determine how to crop
        cropped_inputs[key] = val
  
  return cropped_inputs

def _crop_batch_dict(batch, target_crop_indices, target_len, binder_len):
  """Crop batch dictionary."""
  full_indices = np.concatenate([
    target_crop_indices,
    np.arange(target_len, target_len + binder_len)
  ])
  
  cropped_batch = {}
  for key, val in batch.items():
    if not hasattr(val, 'shape'):
      cropped_batch[key] = val
      continue
      
    # Most batch features are (num_res, ...) or (num_templates, num_res, ...)
    if val.ndim == 1:
      cropped_batch[key] = val[full_indices]
    elif val.ndim == 2:
      if val.shape[0] == target_len + binder_len:
        cropped_batch[key] = val[full_indices, :]
      else:
        cropped_batch[key] = val
    elif val.ndim == 3:
      if val.shape[0] == target_len + binder_len:
        cropped_batch[key] = val[full_indices, :, :]
      elif val.shape[1] == target_len + binder_len:
        cropped_batch[key] = val[:, full_indices, :]
      else:
        cropped_batch[key] = val
    elif val.ndim == 4:
      if val.shape[1] == target_len + binder_len:
        cropped_batch[key] = val[:, full_indices, :, :]
      else:
        cropped_batch[key] = val
    else:
      cropped_batch[key] = val
  
  return cropped_batch

def _crop_prev_dict(prev, target_crop_indices, target_len, binder_len):
  """Crop prev dictionary (prev_pos, prev_msa_first_row, prev_pair)."""
  full_indices = np.concatenate([
    target_crop_indices,
    np.arange(target_len, target_len + binder_len)
  ])
  
  cropped_prev = {}
  for key, val in prev.items():
    if not hasattr(val, 'shape'):
      cropped_prev[key] = val
      continue
      
    if key == 'prev_pos':
      # Shape: (num_res, num_atoms, 3)
      cropped_prev[key] = val[full_indices, :, :]
    elif key == 'prev_msa_first_row':
      # Shape: (num_res, channels)
      cropped_prev[key] = val[full_indices, :]
    elif key == 'prev_pair':
      # Shape: (num_res, num_res, channels)
      cropped_prev[key] = val[np.ix_(full_indices, full_indices)]
    elif key == 'prev_dgram':
      # Shape: (num_res, num_res, bins)
      cropped_prev[key] = val[np.ix_(full_indices, full_indices)]
    else:
      # For unknown keys, try to infer
      if val.ndim == 2 and val.shape[0] == target_len + binder_len:
        cropped_prev[key] = val[full_indices, :]
      elif val.ndim == 3:
        if val.shape[0] == target_len + binder_len and val.shape[1] == target_len + binder_len:
          cropped_prev[key] = val[np.ix_(full_indices, full_indices)]
        elif val.shape[0] == target_len + binder_len:
          cropped_prev[key] = val[full_indices, :, :]
        else:
          cropped_prev[key] = val
      else:
        cropped_prev[key] = val
  
  return cropped_prev

def crop_sizes(input_dict, old_dim, array_slice, verbose=False):
  """Crops the input to the new size given by slice.
  Args:
    input_dict: input dictionary with tensors.
    old_dim: old dimension to be changed.
    array_slice: slice to crop the input tensor to.
    verbose: whether to print debug information.

  Returns:
    Dict of cropped tensors.
  """
  # print('beginning crop dict', old_dim)
  for k in input_dict:
    # print(f'in begin input_dict {k}')
    if isinstance(input_dict[k], dict):
      for kk in input_dict[k]:
        if isinstance(input_dict[k][kk], dict):
          for kkk in input_dict[k][kk]:
            if not hasattr(input_dict[k][kk][kkk], "shape"):
              continue
            indices_to_change = [x for x, y in enumerate(input_dict[k][kk][kkk].shape) if y == old_dim]
            if len(indices_to_change) == 0:
              # print('in input_dict subsubdict nothing to change')
              input_dict[k][kk][kkk] = input_dict[k][kk][kkk]
            else:
              print('STILL DO TO input_dict SUBSUBDICT',indices_to_change)
        else:
          # print(f'input_dict {k}, {kk} shape: {input_dict[k][kk].shape}')
          # if kk == 'crop':
          #   continue
          if not hasattr(input_dict[k][kk], "shape"):
              continue
          try:
            indices_to_change = [x for x, y in enumerate(input_dict[k][kk].shape) if y == old_dim]
          except Exception as e:
            print('in crop_sizes, error',e,k,kk,input_dict[k][kk])
          # print('in input_dict subdict indices',indices_to_change)
          if len(indices_to_change) == 0:
            # print('in input_dict subdict nothing to change')
            input_dict[k][kk] = input_dict[k][kk]
          elif len(indices_to_change) == 1:
            # jax.debug.print(f'verbose in {k}, {kk} indices shape {input_dict[k][kk].shape} subdict')
            # jax.debug.print("vv: {}", input_dict[k][kk])
            if indices_to_change[0] == 0:
              if len(input_dict[k][kk].shape) == 1:
                input_dict[k][kk] = input_dict[k][kk][array_slice]
              elif len(input_dict[k][kk].shape) == 2:
                input_dict[k][kk] = input_dict[k][kk][array_slice, :]
              elif len(input_dict[k][kk].shape) == 3:
                input_dict[k][kk] = input_dict[k][kk][array_slice, :, :]
              elif len(input_dict[k][kk].shape) == 4:
                input_dict[k][kk] = input_dict[k][kk][array_slice, :, :, :]
            elif indices_to_change[0] == 1:
              if len(input_dict[k][kk].shape) == 2:
                input_dict[k][kk] = input_dict[k][kk][:, array_slice]
              elif len(input_dict[k][kk].shape) == 3:
                input_dict[k][kk] = input_dict[k][kk][:, array_slice, :]
              elif len(input_dict[k][kk].shape) == 4:
                input_dict[k][kk] = input_dict[k][kk][:, array_slice, :, :]
            elif indices_to_change[0] == 2:
              if len(input_dict[k][kk].shape) == 3:
                input_dict[k][kk] = input_dict[k][kk][:, :, array_slice]
              elif len(input_dict[k][kk].shape) == 4:
                input_dict[k][kk] = input_dict[k][kk][:, :, array_slice, :]
            elif indices_to_change[0] == 3:
              if len(input_dict[k][kk].shape) == 4:
                input_dict[k][kk] = input_dict[k][kk][:, :, :, array_slice]
          elif len(indices_to_change) == 2:
            if (indices_to_change[0] == 0) & (indices_to_change[1] == 1):
              if len(input_dict[k][kk].shape) == 3:
                input_dict[k][kk] = input_dict[k][kk][array_slice[:, np.newaxis], array_slice,:]
              elif len(input_dict[k][kk].shape) == 2:
                input_dict[k][kk] = input_dict[k][kk][array_slice[:, np.newaxis], array_slice]
              else:
                print('still to do SUBDICT not 3D', input_dict[k][kk].shape)
            else:
              print('still to do SUBDICT with other 2 indices', indices_to_change)
          else:
            print('STILL DO TO input_dict SUBDICT',indices_to_change)
    else:
      if not hasattr(input_dict[k], "shape"):
        input_dict[k] = input_dict[k]
        continue
      indices_to_change = [x for x, y in enumerate(input_dict[k].shape) if y == old_dim]
      # print(f'in input_dict indices, {k}',indices_to_change)
      if len(indices_to_change) == 1:
        # if verbose:
        #   jax.debug.print(f'verbose in input_dict {k} shape: {input_dict[k].shape}')
        #   jax.debug.print("v: {}", input_dict[k])
        if indices_to_change[0] == 0:
          if len(input_dict[k].shape) == 1:
            input_dict[k] = input_dict[k][array_slice]
          elif len(input_dict[k].shape) == 2:
            input_dict[k] = input_dict[k][array_slice, :]
          elif len(input_dict[k].shape) == 3:
            input_dict[k] = input_dict[k][array_slice, :, :]
          elif len(input_dict[k].shape) == 4:
            input_dict[k] = input_dict[k][array_slice, :, :, :]
        if indices_to_change[0] == 1:
          if len(input_dict[k].shape) == 2:
            input_dict[k] = input_dict[k][:, array_slice]
          elif len(input_dict[k].shape) == 3:
            input_dict[k] = input_dict[k][:, array_slice, :]
          elif len(input_dict[k].shape) == 4:
            input_dict[k] = input_dict[k][:, array_slice, :, :]
        if indices_to_change[0] == 2:
          if len(input_dict[k].shape) == 3:
            input_dict[k] = input_dict[k][:, :, array_slice]
          elif len(input_dict[k].shape) == 4:
            input_dict[k] = input_dict[k][:, :, array_slice, :]
        if indices_to_change[0] == 3:
          if len(input_dict[k].shape) == 4:
            input_dict[k] = input_dict[k][:, :, :, array_slice]
      elif len(indices_to_change) > 1:
        if (indices_to_change[0] == 0) & (indices_to_change[1] == 1):
          if len(input_dict[k].shape) == 3:
            input_dict[k] = input_dict[k][array_slice[:, np.newaxis], array_slice,:]
          else:
            print('still to do not 3D in design', k, input_dict[k].shape)
        else:
          print('still to do with more than 2 indices', indices_to_change)
      else:
        # print('nothing to change')
        input_dict[k] = input_dict[k]
  # print('finished crop dict', input_dict)
  return input_dict  

class _af_design:

  def restart(self, seed=None, opt=None, weights=None,
              seq=None, mode=None, keep_history=False, reset_opt=True, **kwargs):   
    '''
    restart the optimization
    ------------
    note: model.restart() resets the [opt]ions and weights to their defaults
    use model.set_opt(..., set_defaults=True) and model.set_weights(..., set_defaults=True)
    or model.restart(reset_opt=False) to avoid this
    ------------
    seed=0 - set seed for reproducibility
    reset_opt=False - do NOT reset [opt]ions/weights to defaults
    keep_history=True - do NOT clear the trajectory/[opt]ions/weights
    '''
    # reset [opt]ions
    if reset_opt and not keep_history:
      copy_missing(self.opt, self._opt)
      self.opt = copy_dict(self._opt)
      if hasattr(self,"aux"): del self.aux
    
    if not keep_history:
      # initialize trajectory
      self._tmp = {"traj":{"seq":[],"xyz":[],"plddt":[],"pae":[]},
                   "log":[],"best":{}}

    # update options/settings (if defined)
    self.set_opt(opt)
    self.set_weights(weights)
  
    # initialize sequence
    self.set_seed(seed)
    self.set_seq(seq=seq, mode=mode, **kwargs)

    # reset optimizer
    self._k = 0
    self.set_optimizer()

  def _get_model_nums(self, num_models=None, sample_models=None, models=None):
    '''decide which model params to use'''
    if num_models is None: num_models = self.opt["num_models"]
    if sample_models is None: sample_models = self.opt["sample_models"]

    ns_name = self._model_names
    ns = list(range(len(ns_name)))
    if models is not None:
      models = models if isinstance(models,list) else [models]
      ns = [ns[n if isinstance(n,int) else ns_name.index(n)] for n in models]

    m = min(num_models,len(ns))
    if sample_models and m != len(ns):
      model_nums = np.random.choice(ns,(m,),replace=False)
    else:
      model_nums = ns[:m]
    return model_nums   

  def run(self, num_recycles=None, num_models=None, sample_models=None, models=None,
          backprop=True, callback=None, model_nums=None, return_aux=False):
    '''run model to get outputs, losses and gradients'''
    
    # pre-design callbacks
    for fn in self._callbacks["design"]["pre"]: fn(self)

    # decide which model params to use
    if model_nums is None:
      model_nums = self._get_model_nums(num_models, sample_models, models)
    assert len(model_nums) > 0, "ERROR: no model params defined"

    # print('in design run beginning, crop', self._cfg.model.embeddings_and_evoformer.crop)
    # loop through model params
    start_time = time.time()
    auxs = []
    for n in model_nums:
      p = self._model_params[n]
      auxs.append(self._recycle(p, num_recycles=num_recycles, backprop=backprop))
    run_time = time.time() - start_time
    # print('in design num_recycles', num_recycles, 'model_nums', model_nums)
    # print('in design, nb auxs',len(auxs))
    # print('in design, auxs',auxs[-1]['pae'].shape)
    # np.save('/scicore/home/schwede/follon0000/BindCraft/example/PBP2A_sepchains_predict/auxs_pae_before.npy', auxs[-1]['pae'])
    auxs = jax.tree_util.tree_map(lambda *x: np.stack(x), *auxs)
    # if not self._cfg.model.embeddings_and_evoformer.crop:
    #   print('in design run, not cropping', auxs['atom_positions'].shape, auxs['seq_pseudo'].shape)
    # #   jax.debug.print('auxs: {}',auxs['atom_positions'].shape)  
    # else:
    #   print('in design run, opt crop is true')

    # update aux (average outputs)
    def avg_or_first(x):
      if np.issubdtype(x.dtype, np.integer): return x[0]
      else: return x.mean(0)

    self.aux = jax.tree_util.tree_map(avg_or_first, auxs)
    # print('in design, aux shape', self.aux["atom_positions"][0].shape)
    # print('in design, saving aux outputs pae',auxs['pae'].shape)
    # np.save('/scicore/home/schwede/follon0000/BindCraft/example/PBP2A_sepchains_predict/auxs_pae.npy', auxs['pae'])
    # if hasattr(self._tmp,"best"):
    #   if hasattr(self._tmp["best"],"aux"):
    #     print('in design, best aux shape', self._tmp["best"]["aux"]["atom_positions"][0].shape)
    #   else:
    #     print('in design, no best aux', self._tmp["best"].keys())
    self.aux["atom_positions"] = auxs["atom_positions"][0]
    self.aux["all"] = auxs
    
    # post-design callbacks
    for fn in (self._callbacks["design"]["post"] + to_list(callback)): fn(self)

    # update log
    self.aux["log"] = {**self.aux["losses"]}
    # if 'sc_rmsd' in self.aux["log"]:
    #   print('in design run, sc_rmsd loss', self.aux["log"]["sc_rmsd"])
    # if 'plddt' not in self.aux["log"]:
    #   print('plddt not in log')
    #   print(self.aux["log"])
    # else:
    #   print('plddt in log')
    #   print('plddt shape',self.aux["log"]["plddt"].shape,self.aux["log"]["plddt"])
    #   jax.debug.print('plddt in log: {}',self.aux["log"]["plddt"])
    self.aux["log"]["plddt"] = 1 - self.aux["log"]["plddt"]
    if "ipsae" in self.aux["log"]:
      self.aux["log"]["ipsae"] = 1 - self.aux["log"]["ipsae"]
    else:
      self.aux["log"]["ipsae"] = 0
    for k in ["loss","i_ptm","ptm"]: self.aux["log"][k] = self.aux[k]
    for k in ["hard","soft","temp"]: self.aux["log"][k] = self.opt[k]

    # compute sequence recovery
    if self.protocol in ["fixbb","partial"] or (self.protocol == "binder" and self._args["redesign"]):
      if self.protocol == "partial":
        aatype = self.aux["aatype"][...,self.opt["pos"]]
      else:
        aatype = self.aux["seq"]["pseudo"].argmax(-1)

      mask = self._wt_aatype != -1
      true = self._wt_aatype[mask]
      pred = aatype[...,mask]
      self.aux["log"]["seqid"] = (true == pred).mean()

    self.aux["log"]["run_time"] = run_time
    self.aux["log"] = to_float(self.aux["log"])
    self.aux["log"].update({"recycles":int(self.aux["num_recycles"]),
                            "models":model_nums})
    # print('in design run, aux keys', self.aux.keys())
    # print('in run, representations in aux', 'representations' in self.aux, 'representations in inputs', 'representations' in self._inputs)
    if return_aux: return self.aux

  def _single(self, model_params, backprop=True):
    '''single pass through the model'''
    self._inputs["opt"] = self.opt
    
    # Pass crop settings to the model if they exist
    if self.protocol == "binder" and self._args.get("crop_target_residues") is not None:
      crop_residues = self._args["crop_target_residues"]
      if crop_residues and len(crop_residues) > 0:
        self._inputs["crop_target_residues"] = crop_residues
        self._inputs["target_len"] = self._target_len
      else:
        self._inputs["crop_target_residues"] = None
    else:
      self._inputs["crop_target_residues"] = None
    # if self._inputs['msa_feat'].shape[1] > self._inputs['representations']['evoformer_input_msa'].shape[1]: #'representations' in self._inputs and 
    #   print('cropping in _single here before running model', self._inputs['msa_feat'].shape, self._inputs['representations']['evoformer_input_msa'].shape)
    #   # jax_array_slice = jnp.concatenate([jnp.arange(35,50),jnp.arange(92,111),jnp.arange(115,self._inputs['msa_feat'].shape[1])], axis=0)
    #   # jax_array_slice = jnp.concatenate([jnp.arange(31,59),jnp.arange(92,111),jnp.arange(115,self._inputs['msa_feat'].shape[1])], axis=0)
    #   crop_indices = self._cfg.model.embeddings_and_evoformer.crop_indices
    #   to_concat = []
    #   for i in range(0, len(crop_indices) - 1, 2):
    #     to_concat.append(jnp.arange(crop_indices[i], crop_indices[i+1]))
    #   to_concat.append(jnp.arange(crop_indices[-1], self._inputs['msa_feat'].shape[1]))
    #   jax_array_slice = jnp.concatenate(to_concat, axis=0)
    #   self._inputs = crop_sizes(self._inputs, self._inputs['msa_feat'].shape[1], jax_array_slice)
    #   self._cfg.model.embeddings_and_evoformer.crop = False
    #   self._cfg.model.embeddings_and_evoformer.crop_indices = []
    #   self.opt['crop'] = False

    # print('in _single, inputs cfg cropping', self._cfg.model.embeddings_and_evoformer.crop)
    flags  = [self._params, model_params, self._inputs, self.key()]
    if backprop:
      # print('running _single with backprop')
      (loss, aux), grad = self._model["grad_fn"](*flags)
    else:
      # print('running _single without backprop')
      loss, aux = self._model["fn"](*flags)
      grad = jax.tree_util.tree_map(np.zeros_like, self._params)
    
    # print('in _single, aux keys', aux['representations']['evoformer_input_msa'].shape, aux['representations']['evoformer_input_pair'].shape, aux['representations']['evoformer_masks_msa'].shape, aux['representations']['evoformer_masks_pair'].shape)
    # # TEMP DEBUG: dump per-frame atom positions as PDBs.
    # # Uses all_stage_atom_positions (S,L,37,3) when available, otherwise
    # # falls back to atom_positions (L,37,3) for a single frame.
    # if "aatype" in aux and "residue_index" in aux:
    #   debug_dir = "/scicore/home/schwede/follon0000/FastCraft/example/tmp_debug/colabdesign_aux_atom_positions"
    #   os.makedirs(debug_dir, exist_ok=True)
    #   if "debug_aux_pdb_n" not in self._tmp:
    #     self._tmp["debug_aux_pdb_n"] = 0
    #
    #   if "all_stage_atom_positions" in aux and "all_stage_atom_mask" in aux:
    #     all_pos = np.asarray(aux["all_stage_atom_positions"])
    #     all_mask = np.asarray(aux["all_stage_atom_mask"])
    #     print(f'in _single, dumping {all_pos.shape[0]} stage PDBs, shape {all_pos.shape}')
    #     for i in range(all_pos.shape[0]):
    #       self._tmp["debug_aux_pdb_n"] += 1
    #       pdb_path = os.path.join(
    #           debug_dir,
    #           f"auxpdb_k{self._k:04d}_n{self._tmp['debug_aux_pdb_n']:06d}_frame{i:02d}.pdb")
    #       self.save_pdb_from_aux(
    #           aux={
    #               "aatype": aux["aatype"],
    #               "residue_index": aux["residue_index"],
    #               "atom_positions": all_pos[i],
    #               "atom_mask": all_mask[i],
    #           },
    #           filename=pdb_path,
    #           renum_pdb=True,
    #       )
    #   elif "atom_positions" in aux and "atom_mask" in aux:
    #     self._tmp["debug_aux_pdb_n"] += 1
    #     pdb_path = os.path.join(
    #         debug_dir,
    #         f"auxpdb_k{self._k:04d}_n{self._tmp['debug_aux_pdb_n']:06d}.pdb")
    #     self.save_pdb_from_aux(
    #         aux={
    #             "aatype": aux["aatype"],
    #             "residue_index": aux["residue_index"],
    #             "atom_positions": aux["atom_positions"],
    #             "atom_mask": aux["atom_mask"],
    #         },
    #         filename=pdb_path,
    #         renum_pdb=True,
    #     )
    # print(stop)
    aux.update({"loss":loss,"grad":grad}) # "representations": aux["representations"]
    # print('in _single, at end crop', self.opt['crop'])
    return aux

  def _recycle(self, model_params, num_recycles=None, backprop=True):   
    '''multiple passes through the model (aka recycle)'''
    a = self._args
    mode = a["recycle_mode"]
    # print('in _recycle, mode', mode)
    if num_recycles is None:
      num_recycles = self.opt["num_recycles"]

    # print('in beginnning of _recycle, inputs and prev', 'prev' in self._inputs, a['clear_prev'])
    if mode in ["backprop","add_prev"]:
      # recycles compiled into model, only need single-pass
      # print('in _recycle, mode backprop or add_prev', mode)
      aux = self._single(model_params, backprop)
    
    else:
      # print('in _recycle, mode else', mode)
      L = self._inputs["residue_index"].shape[0]
      # print('in _recycle, L', L)
      
      # intialize previous
      # print('in _recycle, before if', 'prev' in self._inputs, a["clear_prev"])
      if "prev" not in self._inputs or a["clear_prev"]:
        prev = {'prev_msa_first_row': np.zeros([L,256]),
                'prev_pair': np.zeros([L,L,128])}

        if a["use_initial_guess"]:
          if "batch" in self._inputs:
            # print('in _recycle, batch["all_atom_positions"]', self._inputs["batch"]["all_atom_positions"].shape, self._inputs["batch"]["all_atom_positions"][0])
            if self.opt["crop"]: #self._cfg.model.embeddings_and_evoformer.crop:
              # print('in _recycle, using prev_pos in prev, cropping True')
              # print('in _recycle, use_initial_guess and batch in inputs and cropping', self._inputs["batch"]["all_atom_positions"][35])
              if 'prev_pos' not in prev:
                prev["prev_pos"] = np.zeros([L,37,3])
              # print('in _recycle, prev_pos, cropping True', prev["prev_pos"].shape, self._inputs["batch"]["all_atom_positions"].shape)
              # print('in _recycle, all_atom_positions', self._inputs["batch"]["all_atom_positions"][:34,:,:])
              # print('in _recycle, all_atom_positions', self._inputs["batch"]["all_atom_positions"][:115,:,:])
              # prev["prev_pos"][:self._init_target_len,:,:] = self._inputs["batch"]["all_atom_positions"][:self._init_target_len,:,:] # 494 494  
              prev["prev_pos"] = self._inputs["batch"]["all_atom_positions"] # 494 494
            else:
              # print('in _recycle, using prev_pos in prev, cropping False')
              # print('in _recycle, use_initial_guess and batch in inputs and NOT cropping', self._inputs["batch"]["all_atom_positions"][0])
              # print('in _recycle, prev_pos, cropping False', prev["prev_pos"].shape, self._inputs["batch"]["all_atom_positions"].shape)
              # print('in _recycle, use_initial_guess and batch in inputs and not cropping', self._inputs["batch"]["all_atom_positions"].shape)
              prev["prev_pos"] = self._inputs["batch"]["all_atom_positions"]
            # print('in _recycle, using all_atom_positions', self._inputs["batch"]["all_atom_positions"].shape, self._inputs["batch"]["all_atom_positions"][0])
            # print('in recycle, prev_pos', prev["prev_pos"].shape)
            # print('in _recycle, rm_template_seq', self._inputs['rm_template_seq'])
            # print('in _recycle, rm_template_sc', self._inputs['rm_template_sc'])
            # print('in _recycle, _target_len', self._target_len)
        else:
          # print('in _recycle, not using initial guess')
          prev["prev_pos"] = np.zeros([L,37,3])

        if a["use_dgram"]:
          # TODO: add support for initial_guess + use_dgram
          prev["prev_dgram"] = np.zeros([L,L,64])

        if a["use_initial_atom_pos"]:
          if "batch" in self._inputs:
            # self._inputs["initial_atom_pos"] = self._inputs["batch"]["all_atom_positions"]  #OLD
            initial_pos = np.copy(self._inputs["batch"]["all_atom_positions"])
            # TMP: fill binder (zero) coordinates with first hotspot's atoms
            # so the structure module gets a valid rigid frame instead of a
            # degenerate zero matrix that absorbs all fold-iteration updates.
            hotspot_idx = None
            if "hotspot" in self.opt and self.opt["hotspot"] is not None and len(self.opt["hotspot"]) > 0:
              hotspot_idx = int(self.opt["hotspot"][0])
            elif "crop_target_hotspot_residues" in self.opt and self.opt["crop_target_hotspot_residues"] is not None and len(self.opt["crop_target_hotspot_residues"]) > 0:
              hotspot_idx = int(self.opt["crop_target_hotspot_residues"][0])
            if hotspot_idx is not None and hotspot_idx < initial_pos.shape[0]:
              hotspot_coords = initial_pos[hotspot_idx]
              initial_pos[self._target_len:] = hotspot_coords[None]
            #   print(f'in _recycle, initial_atom_pos: filled binder with hotspot {hotspot_idx} coords, shape {initial_pos.shape}')
            # else:
            #   print(f'in _recycle, initial_atom_pos: no hotspot found, binder stays zero, shape {initial_pos.shape}')
            self._inputs["initial_atom_pos"] = initial_pos
          else:
            # print('in _recycle, not using initial atom pos because no batch in inputs')
            self._inputs["initial_atom_pos"] = np.zeros([L,37,3])

        # if 'representations' in self._inputs:
        #   print('in _recycle, adding representations to prev', self._inputs['prev'].keys())
        #   prev["prev_evoformer_input_msa"] = self._inputs['prev']["prev_evoformer_input_msa"]
        #   prev["prev_evoformer_input_pair"] = self._inputs['prev']["prev_evoformer_input_pair"]
        #   prev["prev_evoformer_masks_msa"] = self._inputs['prev']["prev_evoformer_masks_msa"]
        #   prev["prev_evoformer_masks_pair"] = self._inputs['prev']["prev_evoformer_masks_pair"]
      
        self._inputs["prev"] = prev
      # decide which layers to compute gradients for
      cycles = (num_recycles + 1)
      mask = [0] * cycles

      if mode == "sample":  mask[np.random.randint(0,cycles)] = 1
      if mode == "average": mask = [1/cycles] * cycles
      if mode == "last":    mask[-1] = 1
      if mode == "first":   mask[0] = 1
      if mode == "none":    mask = mask
      # print('in _recycle, mask', mask)

      # gather gradients across recycles 
      grad = []
      for m in mask:      
        # print('in recycle iterations', m, 'crop', self.opt['crop'], self._cfg.model.embeddings_and_evoformer.crop, self._target_len)  
        # if 'prev' in self._inputs:
        #   print('in _recycle before single, prev in inputs')
        #   if 'prev_pos' in self._inputs["prev"]:
        #     print('in _recycle before single, self._inputs["prev_pos"].shape', self._inputs["prev"]["prev_pos"].shape, self._inputs["prev"]["prev_pos"][0])
        #   else:
        #     print('in _recycle before single, no prev_pos in inputs')
        if m == 0:
          aux = self._single(model_params, backprop=False)
        else:
          aux = self._single(model_params, backprop)
          grad.append(jax.tree_util.tree_map(lambda x:x*m, aux["grad"]))
        # print('in _recycle, prev_evoformer_input_msa in aux prev', 'prev_evoformer_input_msa' in aux['prev'])
        # print('in _recycle before update, prev_pos', aux["prev"]["prev_pos"].shape, self._inputs["prev"]["prev_pos"].shape)
        # print('in _recycle, aux atom_positions', aux["atom_positions"].shape, aux["atom_positions"][0][0])
        # print('in _recycle, aux["prev"]["prev_pos"]',  aux["prev"]["prev_pos"].shape,  aux["prev"]["prev_pos"][0][0])
        # print('in _recycle aux', aux.keys())
        self._inputs["prev"] = aux["prev"]

        # self._inputs["representations"] = aux["representations"]
        # print('in _recycle, aux["prev"]["prev_pos"].shape', aux["prev"]["prev_pos"].shape, aux["prev"]["prev_pos"][0])
        # print('in _recycle, aux["atom_positions"]', aux["atom_positions"].shape, aux["atom_positions"][0])
        if a["use_initial_atom_pos"]:
          # print('in _recycle, using initial atom pos', aux["prev"]["prev_pos"].shape)
          self._inputs["initial_atom_pos"] = aux["prev"]["prev_pos"]    
        # print('in _recycle, at end of single recycle, initial_atom_pos', aux["prev"]["prev_pos"] .shape)
        # print('in _recycle, at end of single recycle, crop', self._cfg.model.embeddings_and_evoformer.crop) #self.opt['crop']
      if not mode == "none":
        aux["grad"] = jax.tree_util.tree_map(lambda *x: np.stack(x).sum(0), *grad)
    
    aux["num_recycles"] = num_recycles
    # if 'evoformer_input_msa' in self._inputs["representations"]:
    #   print('in _recycle, popping representations', self._inputs["representations"]["evoformer_input_msa"].shape)
    #   # print('in _recycle, popping representations', self._inputs["representations"]['evoformer_input_msa'].shape)
    #   self._inputs["representations"].pop('evoformer_input_msa')
    #   self._inputs["representations"].pop('evoformer_input_pair')
    #   self._inputs["representations"].pop('evoformer_masks_msa')
    #   self._inputs["representations"].pop('evoformer_masks_pair')
    # if 'evoformer_input_msa' in aux["representations"]:
    #   print('removing from aux')
    #   aux["representations"].pop('evoformer_input_msa')
    #   aux["representations"].pop('evoformer_input_pair')
    #   aux["representations"].pop('evoformer_masks_msa')
    #   aux["representations"].pop('evoformer_masks_pair')
    # print('at end of _recycle, representations in aux', 'representations' in aux, 'representations in inputs', 'representations' in self._inputs)
    return aux

  def step(self, lr_scale=1.0, num_recycles=None,
           num_models=None, sample_models=None, models=None, backprop=True,
           callback=None, save_best=False, verbose=1, save_pdb=False, id=''):
    '''do one step of gradient descent'''
    
    # run
    self.run(num_recycles=num_recycles, num_models=num_models, sample_models=sample_models,
             models=models, backprop=backprop, callback=callback)

    # modify gradients    
    if self.opt["norm_seq_grad"]: self._norm_seq_grad()
    self._state, self.aux["grad"] = self._optimizer(self._state, self.aux["grad"], self._params)
  
    # print('in step, before applying gradients', self._params['seq'].shape, self._params['seq'])
    # apply gradients
    lr = self.opt["learning_rate"] * lr_scale
    self._params = jax.tree_util.tree_map(lambda x,g:x-lr*g, self._params, self.aux["grad"])
    # print('in step, after applying gradients', self._params['seq'].shape, self._params['seq'])

    # save results
    # print('in step, saving results', save_best)
    self._save_results(save_best=save_best, verbose=verbose)
    # print('in step, saving first pdb')
    start_time = time.time()
    if save_pdb and id:
      # Save PDB with iteration number (1, 11, 21, ..., or last iteration)
      pdb_path = id.replace(f"{id.split('/')[-1]}", f"{id.split('/')[-1].split('.')[0]}_{self._k+1}.pdb") #f'/scicore/home/schwede/follon0000/FastCraft/comp/CDR_design_0recycle/Trajectory/{id}_{self._k+1}.pdb'
      # print(f'Saving iteration {self._k+1} to {pdb_path}')
      self.save_pdb(pdb_path)
      # print('in step, end, self._inputs[rm_template)',self._inputs['rm_template'])
      # print('in step, end, self._inputs[rm_template_seq)',self._inputs['rm_template_seq'])
      # print('in step, end, self._inputs[rm_template_sc)',self._inputs['rm_template_sc'])

      # print('decision to repredict', self._cfg.model.embeddings_and_evoformer.crop, self.opt["num_recycles"])
      if self.opt["num_recycles"] == 0: #self._cfg.model.embeddings_and_evoformer.crop and 
        # print('in step, REPREDICTING!!')
        # print('is aux in tmp?', 'aux' in self._tmp,self._tmp.keys())
        # print('is seq in aux?', 'aux' in self.aux, self.aux.keys())
        if 'seq' in self.aux: #'aux' _tmp
          # print('in step, aux in tmp')
          seq_final = self.aux["seq"]["logits"].argmax(-1)
        else:
          seq_final = None
        # print('in step, seq_final', seq_final, len(seq_final[0]))
        # seq_final = self._tmp["traj"]["seq"].argmax(-1)
        tmp_crop = self.opt["crop"] #self._cfg.model.embeddings_and_evoformer.crop
        tmp_recycle_mode = self._args['recycle_mode']
        tmp_use_initial_guess = self._args['use_initial_guess']
        tmp_target_len = self._target_len
        tmp_lengths = self._lengths
        tmp_num_recycles = self.opt["num_recycles"]

        # self._cfg.model.embeddings_and_evoformer.crop = False
        self.opt["crop"] = False
        tmp_model = self._model
        self._model = self._get_model(self._cfg)
        # self._args['recycle_mode'] = 'none'
        self._args['use_initial_guess'] = True
        self.opt["num_recycles"] = 1
        # self._target_len = 115
        # print('in step repredicting, target_len', self._target_len, 'cropping', self._cfg.model.embeddings_and_evoformer.crop)
        if not self.opt["crop"] and self._target_len != self._init_target_len:
          add_length = self._init_target_len - self._target_len #81 #219
          # print('in step, adding length', add_length)
        else:
          add_length = 0
        # print('in step repredicting, add_length', add_length)
        self._target_len = self._target_len + add_length #219 #self._cfg.model.embeddings_and_evoformer.target_len
        self._lengths = [self._lengths[0] + add_length, self._lengths[1]] 
        if "aux" in self._tmp["best"]: 
          tmp_inputs = copy_dict(self._inputs)
          ## NEW
          for key in ("rm_template", "rm_template_seq", "rm_template_sc"):
            if key in self._inputs and isinstance(self._inputs[key], np.ndarray):
                tmp_inputs[key] = self._inputs[key].copy()
          if "batch" in self._inputs and self._inputs["batch"] is not None:
              tmp_inputs["batch"] = jax.tree_util.tree_map(np.array, self._inputs["batch"])
          ## END NEW
          # print('in step, aux in tmp, size rm_template', self._inputs['rm_template'].shape)
          self._inputs['rm_template'][:] = False
          self._inputs['rm_template_seq'][:] = False
          self._inputs['rm_template_sc'][self._target_len:] = False #494
          self._inputs['rm_template_sc'][:self._target_len] = True #494
          self._update_template(self._inputs, self.key())
        else:
          print('todo: rescoring')
          print(stop)
        # print('in step, repredicting with num_recycles = 1')
        # print('no running predict')
        # self.predict(seq=seq_final, num_recycles=num_recycles, models=models, num_models=num_models, sample_models=sample_models, save_final=False, id=id, save_pdb=True)
        # self._cfg.model.embeddings_and_evoformer.crop = tmp_crop
        self.opt["crop"] = tmp_crop
        self._model = self._get_model(self._cfg)
        self._args['recycle_mode'] = tmp_recycle_mode
        self._args['use_initial_guess'] = tmp_use_initial_guess
        self.opt["num_recycles"] = tmp_num_recycles
        self._inputs = tmp_inputs
        self._target_len = tmp_target_len
        self._lengths = tmp_lengths
        # print('with update template')
        self._update_template(self._inputs, self.key()) #NEW!!
        # print('using original model')
        self._model = tmp_model
        # print('in step, after repredicting, crop', self._cfg.model.embeddings_and_evoformer.crop)
    # print('in step, repredict time added', time.time() - start_time)
    # increment
    self._k += 1
    # if self._target_len != 115:
    #   print('in _recycle, resetting current length', self._target_len)
    #   add_length = 81 #219
    #   self._target_len = self._target_len + add_length #219 #self._cfg.model.embeddings_and_evoformer.target_len
    #   self._lengths = [self._lengths[0] + add_length, self._lengths[1]] 
    # self._cfg.model.embeddings_and_evoformer.crop = True
    # print('at end of step, representations present', 'representations' in self._inputs)

  def _print_log(self, print_str=None, aux=None):
    if aux is None: aux = self.aux
    keys = ["models","recycles","hard","soft","temp","seqid","loss",
            "seq_ent","mlm","helix","pae","i_pae","exp_res","con","i_con",
            "sc_fape","sc_rmsd","dgram_cce","fape","plddt","ptm","ipsae","time"]
    
    if "i_ptm" in aux["log"]:
      if len(self._lengths) > 1:
        keys.append("i_ptm")
      else:
        aux["log"].pop("i_ptm")
    # if "ipsae" in aux["log"]:
    #   print('in _print_log, ipsae', aux["log"]["ipsae"])
    # else:
    #   print('in _print_log, ipsae not in aux["log"]')
    print(dict_to_str(aux["log"], filt=self.opt["weights"],
                      print_str=print_str, keys=keys+["rmsd"], ok=["plddt","rmsd"]))

  def _save_results(self, aux=None, save_best=False,
                    best_metric=None, metric_higher_better=False,
                    verbose=True, final=False):
    if aux is None: aux = self.aux    
    self._tmp["log"].append(aux["log"])    
    if (self._k % self._args["traj_iter"]) == 0:
      # update traj
      traj = {"seq":   aux["seq"]["pseudo"],
              "xyz":   aux["atom_positions"][:,1,:],
              "plddt": aux["plddt"],
              "pae":   aux["pae"]}
      for k,v in traj.items():
        if len(self._tmp["traj"][k]) == self._args["traj_max"]:
          self._tmp["traj"][k].pop(0)
        self._tmp["traj"][k].append(v)

    # save best
    if save_best:
      if best_metric is None:
        best_metric = self._args["best_metric"]
      metric = float(aux["log"][best_metric])
      if self._args["best_metric"] in ["plddt","ptm","i_ptm","seqid","composite"] or metric_higher_better:
        metric = -metric
      if final or "metric" not in self._tmp["best"] or metric < self._tmp["best"]["metric"]: #not self._cfg.model.embeddings_and_evoformer.crop or 
        # print('in save best, saving new best')
        # print('last positions', aux["atom_positions"][-1, :, :])
        self._tmp["best"]["aux"] = copy_dict(aux)
        self._tmp["best"]["metric"] = metric
      # else:
      #   print('in save best, not saving new best')

    if verbose and ((self._k+1) % verbose) == 0:
      self._print_log(f"{self._k+1}", aux=aux)

  def predict(self, seq=None, bias=None,
              num_models=None, num_recycles=None, models=None, sample_models=False,
              dropout=False, hard=True, soft=False, temp=1,
              return_aux=False, verbose=True,  seed=None, save_final=False, save_pdb=False, id='', **kwargs):
    '''predict structure for input sequence (if provided)'''

    def load_settings():    
      if "save" in self._tmp and "opt" in self._tmp["save"][3]:
        [self.opt, self._args, self._params, self._inputs] = self._tmp.pop("save")

    def save_settings():
      load_settings()
      self._tmp["save"] = [copy_dict(x) for x in [self.opt, self._args, self._params, self._inputs]]

    if save_final:
      #update opt
      # self.opt['crop'] = self._cfg.model.embeddings_and_evoformer.crop
      if not self.opt["crop"] and self._target_len != self._init_target_len: #self._cfg.model.embeddings_and_evoformer.crop
        # print('in predict save final, adding length', self._target_len)
        add_length = self._init_target_len - self._target_len #81 #219
      else:
        add_length = 0
        # print('in predict save final, not adding length')

      # print('in predict save final, add_length', add_length, self._target_len)
      self._target_len = self._target_len + add_length #219 #self._cfg.model.embeddings_and_evoformer.target_len
      self._len = self._len #self._cfg.model.embeddings_and_evoformer.len
      self._lengths = [self._lengths[0] + add_length, self._lengths[1]] 
      # print('in predict save final, after update opt', self.opt['crop'])
      # print('in predict save final, after update opt len', self._target_len, self._binder_len, self._len)
      # print('in predict save final, self._params["seq"]', self._params["seq"], self._params["seq"].shape)

    save_settings()

    # set seed if defined
    if seed is not None: self.set_seed(seed)

    # set [seq]uence/[opt]ions
    # print('in predict, seq', seq, bias)
    if seq is not None: self.set_seq(seq=seq, bias=bias)    
    self.set_opt(hard=hard, soft=soft, temp=temp, dropout=dropout, pssm_hard=True)
    self.set_args(shuffle_first=False)
    
    # run
    self.run(num_recycles=num_recycles, num_models=num_models,
             sample_models=sample_models, models=models, backprop=False, **kwargs)
    if verbose: self._print_log("predict")

    load_settings()

    start_time = time.time()
    # save results
    if save_final:
      self._save_results(save_best=True, verbose=verbose, final=save_final)
    if save_pdb and id:
      # Save PDB with _1recycle suffix for this iteration
      pdb_path = id.replace(f"{id.split('/')[-1]}", f"{id.split('/')[-1].split('.')[0]}_{self._k+1}_1recycle.pdb")
      # print(f'Saving predict iteration {self._k+1} to {pdb_path}')
      self.save_pdb(pdb_path, get_best=False)
    # print('predict extra time for saving', time.time() - start_time)

    # return (or save) results
    if return_aux: return self.aux

  # ---------------------------------------------------------------------------------
  # example design functions
  # ---------------------------------------------------------------------------------
  def design(self, iters=100,
             soft=0.0, e_soft=None,
             temp=1.0, e_temp=None,
             hard=0.0, e_hard=None,
             step=1.0, e_step=None,
             dropout=True, opt=None, weights=None, 
             num_recycles=None, ramp_recycles=False, 
             num_models=None, sample_models=None, models=None,traj_id='',
             backprop=True, callback=None, save_best=False, verbose=1):

    # update options/settings (if defined)
    self.set_opt(opt, dropout=dropout)
    self.set_weights(weights)    
    m = {"soft":[soft,e_soft],"temp":[temp,e_temp],
         "hard":[hard,e_hard],"step":[step,e_step]}
    m = {k:[s,(s if e is None else e)] for k,(s,e) in m.items()}

    if ramp_recycles:
      if num_recycles is None:
        num_recycles = self.opt["num_recycles"]
      m["num_recycles"] = [0,num_recycles]

    for i in range(iters):
      # print('in design, iter', i)
      for k,(s,e) in m.items():
        if k == "temp":
          self.set_opt({k:(e+(s-e)*(1-(i+1)/iters)**2)})
        else:
          v = (s+(e-s)*((i+1)/iters))
          if k == "step": step = v
          elif k == "num_recycles": num_recycles = round(v)
          else: self.set_opt({k:v})
      
      # decay learning rate based on temperature
      lr_scale = step * ((1 - self.opt["soft"]) + (self.opt["soft"] * self.opt["temp"]))
      
      # Save PDB every 10th iteration
      save_pdb = (i % 10 == 0) or (i == iters - 1)
      
      # print('before step, num_recycles', num_recycles, 'in opt', self.opt["num_recycles"])
      # print('in design before step, representations present', 'representations' in self._inputs)
      self.step(lr_scale=lr_scale, num_recycles=num_recycles,
                num_models=num_models, sample_models=sample_models, models=models,
                backprop=backprop, callback=callback, save_best=save_best, verbose=verbose, save_pdb=save_pdb,id=traj_id)

  def design_logits(self, iters=100, **kwargs):
    ''' optimize logits '''
    self.design(iters, **kwargs)

  def design_soft(self, iters=100, temp=1, **kwargs):
    ''' optimize softmax(logits/temp)'''
    self.design(iters, soft=1, temp=temp, **kwargs)
  
  def design_hard(self, iters=100, **kwargs):
    ''' optimize argmax(logits) '''
    self.design(iters, soft=1, hard=1, **kwargs)

  # ---------------------------------------------------------------------------------
  # experimental
  # ---------------------------------------------------------------------------------
  def design_3stage(self, soft_iters=300, temp_iters=100, hard_iters=10,
                    ramp_recycles=True, **kwargs):
    '''three stage design (logits→soft→hard)'''

    verbose = kwargs.get("verbose",1)

    # stage 1: logits -> softmax(logits/1.0)
    if soft_iters > 0:
      if verbose: print("Stage 1: running (logits → soft)")
      self.design_logits(soft_iters, e_soft=1,
        ramp_recycles=ramp_recycles, **kwargs)
      self._tmp["seq_logits"] = self.aux["seq"]["logits"]
      
    # stage 2: softmax(logits/1.0) -> softmax(logits/0.01)
    if temp_iters > 0:
      if verbose: print("Stage 2: running (soft → hard)")
      self.design_soft(temp_iters, e_temp=1e-2, **kwargs)
    
    # stage 3:
    if hard_iters > 0:
      if verbose: print("Stage 3: running (hard)")
      kwargs["dropout"] = False
      kwargs["save_best"] = True
      kwargs["num_models"] = len(self._model_names)
      self.design_hard(hard_iters, temp=1e-2, **kwargs)

  def _mutate(self, seq, plddt=None, logits=None, mutation_rate=1):
    '''mutate random position'''
    seq = np.array(seq)
    N,L = seq.shape
    # print(f'[DEBUG _mutate] seq shape: {seq.shape}, L={L}')

    # fix some positions
    # Handle case where plddt comes from cropped prediction (size mismatch with full seq)
    if plddt is not None:
      plddt_array = np.maximum(1-plddt,0)
      # print(f'[DEBUG _mutate] plddt length: {len(plddt_array)}')
      if len(plddt_array) != L:
        # plddt size doesn't match sequence (e.g., from cropped prediction)
        # Use uniform probabilities instead of plddt-based weighting
        # print(f'[DEBUG _mutate] plddt size mismatch: {len(plddt_array)} != {L}, using uniform probabilities')
        i_prob = np.ones(L)
      else:
        i_prob = plddt_array
    else:
      i_prob = np.ones(L)
    
    i_prob[np.isnan(i_prob)] = 0
    i_prob_len = len(i_prob)
    # print(f'[DEBUG _mutate] i_prob length: {i_prob_len}')
    
    if "fix_pos" in self.opt:
      # print(f'[DEBUG _mutate] fix_pos in self.opt')
      if "pos" in self.opt:
        # print(f'[DEBUG _mutate] pos in self.opt, len={len(self.opt["pos"])}, first 10: {self.opt["pos"][:10] if len(self.opt["pos"]) >= 10 else self.opt["pos"]}')
        # print(f'[DEBUG _mutate] fix_pos len={len(self.opt["fix_pos"])}, first 10: {self.opt["fix_pos"][:10] if len(self.opt["fix_pos"]) >= 10 else self.opt["fix_pos"]}')
        # print(f'[DEBUG _mutate] fix_pos max={self.opt["fix_pos"].max() if len(self.opt["fix_pos"]) > 0 else "N/A"}')
        # Check for out of bounds indices in fix_pos BEFORE indexing into pos
        fix_pos_valid_mask = self.opt["fix_pos"] < len(self.opt["pos"])
        fix_pos_valid = self.opt["fix_pos"][fix_pos_valid_mask]
        # print(f'[DEBUG _mutate] After filtering fix_pos: {len(fix_pos_valid)} valid out of {len(self.opt["fix_pos"])}')
        if len(fix_pos_valid) > 0:
          p = np.array(self.opt["pos"][fix_pos_valid])
          # print(f'[DEBUG _mutate] p (indexed from pos) length: {len(p)}, max: {p.max() if len(p) > 0 else "N/A"}')
          # Filter out indices that exceed sequence length L
          valid_mask = p < L
          p_valid = p[valid_mask]
          # print(f'[DEBUG _mutate] After filtering by L: {len(p_valid)} valid positions (out of {len(p)})')
          if len(p_valid) > 0:
            # Update corresponding mask for _wt_aatype_sub indexing
            valid_mask_updated = np.isin(p, p_valid)
            seq[...,p_valid] = self._wt_aatype_sub[..., valid_mask_updated]
            i_prob[p_valid] = 0
            # print(f'[DEBUG _mutate] Set i_prob to 0 for {len(p_valid)} fixed positions')
            # print(f'[DEBUG _mutate] i_prob sum after fixing: {i_prob.sum()}, non-zero positions: {np.count_nonzero(i_prob)}')
      else:
        # print(f'[DEBUG _mutate] pos NOT in self.opt')
        p = np.array(self.opt["fix_pos"])
        # Filter by sequence length L
        valid_mask = p < L
        p_valid = p[valid_mask]
        if len(p_valid) > 0:
          seq[...,p_valid] = self._wt_aatype[...,p_valid]
          i_prob[p_valid] = 0
          # print(f'[DEBUG _mutate] Set i_prob to 0 for {len(p_valid)} fixed positions')
          # print(f'[DEBUG _mutate] i_prob sum after fixing: {i_prob.sum()}, non-zero positions: {np.count_nonzero(i_prob)}')
    
    for m in range(mutation_rate):
      # sample position
      # https://www.biorxiv.org/content/10.1101/2021.08.24.457549v1
      i = np.random.choice(np.arange(L),p=i_prob/i_prob.sum())

      # sample amino acid
      logits = np.array(0 if logits is None else logits)
      if logits.ndim == 3: logits = logits[:,i]
      elif logits.ndim == 2: logits = logits[i]
      a_logits = logits - np.eye(self._args["alphabet_size"])[seq[:,i]] * 1e8
      a = categorical(softmax(a_logits))

      # print(f'[DEBUG _mutate] sampling amino acid for position {i}, a={a}')
      # return mutant
      seq[:,i] = a
      # print(f'[DEBUG _mutate] seq after mutation: {seq[:,i]}, {seq.shape}')
    return seq

  def design_semigreedy(self, iters=100, tries=10, dropout=False,traj_id='',
                        save_best=True, seq_logits=None, e_tries=None, **kwargs):

    '''semigreedy search'''    
    if e_tries is None: e_tries = tries

    #update opt
    # self.opt['crop'] = self._cfg.model.embeddings_and_evoformer.crop
    # self._target_len = self._target_len + 100 #self._cfg.model.embeddings_and_evoformer.target_len
    # self._len = self._len #self._cfg.model.embeddings_and_evoformer.len
    # self._lengths = [self._lengths[0] + 100, self._lengths[1]]
    # print('in design_semigreedy, after update opt', self.opt['crop'])
    # print('in design_semigreedy, after update opt len', self._target_len, self._binder_len, self._len)
    # print('in design_semigreedy, self._params["seq"]', self._params["seq"], self._params["seq"].shape)


    # get starting sequence
    if hasattr(self,"aux"):
      # print('in design_semigreedy, getting seq from aux')
      seq = self.aux["seq"]["logits"].argmax(-1)
    else:
      # print('in design_semigreedy, getting seq from params and inputs')
      seq = (self._params["seq"] + self._inputs["bias"]).argmax(-1)
    # print('in design_semigreedy, seq', seq, seq.shape)
    # bias sampling towards the defined bias
    if seq_logits is None: seq_logits = 0
    
    model_flags = {k:kwargs.pop(k,None) for k in ["num_models","sample_models","models"]}
    verbose = kwargs.pop("verbose",1)

    # get current plddt
    # print('in design_semigreedy, getting plddt')
    # print('in design_semigreedy, config', self._cfg.model.embeddings_and_evoformer.crop)
    # print('in design_semigreedy, seq', seq, seq.shape)
    aux = self.predict(seq, return_aux=True, verbose=False, **model_flags, **kwargs)
    plddt = self.aux["plddt"]
    plddt = plddt[self._target_len:] if self.protocol == "binder" else plddt[:self._len]

    # optimize!
    if verbose:
      print("Running semigreedy optimization...")
    # print('in design_semigreedy, number of iters', iters)
    for i in range(iters):
      # print('in design_semigreedy, iter', i)
      buff = []
      model_nums = self._get_model_nums(**model_flags)
      num_tries = (tries+(e_tries-tries)*((i+1)/iters))
      for t in range(int(num_tries)):
        # print('in design_semigreedy, mutate', t, 'iters', i)
        mut_seq = self._mutate(seq=seq, plddt=plddt,
                               logits=seq_logits + self._inputs["bias"])
        # print('in design_semigreedy, mutated seq', mut_seq, mut_seq.shape)
        aux = self.predict(seq=mut_seq, return_aux=True, model_nums=model_nums, verbose=False, **kwargs)
        buff.append({"aux":aux, "seq":np.array(mut_seq)})

      # accept best
      losses = [x["aux"]["loss"] for x in buff]
      best = buff[np.argmin(losses)]
      self.aux, seq = best["aux"], jnp.array(best["seq"])
      # print('saving seq', seq, seq.shape)
      self.set_seq(seq=seq, bias=self._inputs["bias"])
      self._save_results(save_best=save_best, verbose=verbose)

      start_time = time.time()
      # Save PDB every 10th iteration or last iteration
      should_save = (i % 10 == 0) or (i == iters - 1)
      if should_save and traj_id:
        pdb_path = traj_id.replace(f"{traj_id.split('/')[-1]}", f"{traj_id.split('/')[-1].split('.')[0]}_{self._k+1}.pdb")
        # print(f'Saving semigreedy iteration {self._k+1} to {pdb_path}')
        self.save_pdb(pdb_path)
      # print('in design_semigreedy, save pdb time', time.time() - start_time)

      # update plddt
      plddt = best["aux"]["plddt"]
      plddt = plddt[self._target_len:] if self.protocol == "binder" else plddt[:self._len]
      self._k += 1

  def design_pssm_semigreedy(self, soft_iters=300, hard_iters=32, tries=10, e_tries=None,traj_id='',
                             ramp_recycles=True, ramp_models=True, **kwargs):

    verbose = kwargs.get("verbose",1)
    # print('in design_pssm_semigreedy, crop_indices', self._cfg.model.embeddings_and_evoformer.crop_indices)

    # stage 1: logits -> softmax(logits)
    if soft_iters > 0:
      self.design_3stage(soft_iters, 0, 0, ramp_recycles=ramp_recycles, traj_id=traj_id, **kwargs)
      self._tmp["seq_logits"] = kwargs["seq_logits"] = self.aux["seq"]["logits"]

    # stage 2: semi_greedy
    if hard_iters > 0:
      kwargs["dropout"] = False
      if ramp_models:
        num_models = len(kwargs.get("models",self._model_names))
        iters = hard_iters
        for m in range(num_models):
          if verbose and m > 0: print(f'Increasing number of models to {m+1}.')

          kwargs["num_models"] = m + 1
          kwargs["save_best"] = (m + 1) == num_models
          self.design_semigreedy(iters, tries=tries, e_tries=e_tries, traj_id=traj_id, **kwargs)
          if m < 2: iters = iters // 2
      else:
        # print('design_semigreedy hard_iters', hard_iters)
        self.design_semigreedy(hard_iters, tries=tries, e_tries=e_tries, traj_id=traj_id, **kwargs)

  # ---------------------------------------------------------------------------------
  # experimental optimizers (not extensively evaluated)
  # ---------------------------------------------------------------------------------

  def _design_mcmc(self, steps=1000, half_life=200, T_init=0.01, mutation_rate=1,
                   seq_logits=None, save_best=True, **kwargs):
    '''
    MCMC with simulated annealing
    ----------------------------------------
    steps = number for steps for the MCMC trajectory
    half_life = half-life for the temperature decay during simulated annealing
    T_init = starting temperature for simulated annealing. Temperature is decayed exponentially
    mutation_rate = number of mutations at each MCMC step
    '''

    # code borrowed from: github.com/bwicky/oligomer_hallucination

    # gather settings
    verbose = kwargs.pop("verbose",1)
    model_flags = {k:kwargs.pop(k,None) for k in ["num_models","sample_models","models"]}

    # initialize
    plddt, best_loss, current_loss = None, np.inf, np.inf 
    current_seq = (self._params["seq"] + self._inputs["bias"]).argmax(-1)
    if seq_logits is None: seq_logits = 0

    # run!
    if verbose: print("Running MCMC with simulated annealing...")
    for i in range(steps):

      # update temperature
      T = T_init * (np.exp(np.log(0.5) / half_life) ** i) 

      # mutate sequence
      if i == 0:
        mut_seq = current_seq
      else:
        mut_seq = self._mutate(seq=current_seq, plddt=plddt,
                               logits=seq_logits + self._inputs["bias"],
                               mutation_rate=mutation_rate)

      # get loss
      model_nums = self._get_model_nums(**model_flags)
      aux = self.predict(seq=mut_seq, return_aux=True, verbose=False, model_nums=model_nums, **kwargs)
      loss = aux["log"]["loss"]
  
      # decide
      delta = loss - current_loss
      if i == 0 or delta < 0 or np.random.uniform() < np.exp( -delta / T):

        # accept
        (current_seq,current_loss) = (mut_seq,loss)
        
        plddt = aux["all"]["plddt"].mean(0)
        plddt = plddt[self._target_len:] if self.protocol == "binder" else plddt[:self._len]
        
        if loss < best_loss:
          (best_loss, self._k) = (loss, i)
          self.set_seq(seq=current_seq, bias=self._inputs["bias"])
          self._save_results(save_best=save_best, verbose=verbose)
