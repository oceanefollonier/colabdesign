import os
import jax
import jax.numpy as jnp
import numpy as np
from inspect import signature

from colabdesign.af.alphafold.model import data, config, model, all_atom

from colabdesign.shared.model import design_model
from colabdesign.shared.utils import Key

from colabdesign.af.prep   import _af_prep
from colabdesign.af.loss   import _af_loss, get_plddt, get_pae, get_ptm
from colabdesign.af.loss   import get_contact_map, get_seq_ent_loss, get_mlm_loss, get_ipsae
from colabdesign.af.utils  import _af_utils
from colabdesign.af.design import _af_design
from colabdesign.af.inputs import _af_inputs, update_seq, update_aatype
import time

################################################################
# MK_DESIGN_MODEL - initialize model, and put it all together
################################################################

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
  for k in input_dict:
    if isinstance(input_dict[k], dict):
      for kk in input_dict[k]:
        if isinstance(input_dict[k][kk], dict):
          for kkk in input_dict[k][kk]:
            indices_to_change = [x for x, y in enumerate(input_dict[k][kk][kkk].shape) if y == old_dim]
            if len(indices_to_change) == 0:
              input_dict[k][kk][kkk] = input_dict[k][kk][kkk]
            else:
              print('STILL DO TO input_dict SUBSUBDICT',indices_to_change)
        else:
          if kk == 'crop' or kk == 'crop_indices' or kk == 'crop_target_hotspot_residues':
            continue
          try:
            indices_to_change = [x for x, y in enumerate(input_dict[k][kk].shape) if y == old_dim]
          except Exception as e:
            print('in crop_sizes, error',e,k,kk,input_dict[k][kk])
          if len(indices_to_change) == 0:
            input_dict[k][kk] = input_dict[k][kk]
          elif len(indices_to_change) == 1:
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
      if len(indices_to_change) == 1:
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
            print('still to do not 3D in model', k, input_dict[k].shape)
        else:
          print('still to do with more than 2 indices', indices_to_change)
      else:
        input_dict[k] = input_dict[k]
  return input_dict  


class mk_af_model(design_model, _af_inputs, _af_loss, _af_prep, _af_design, _af_utils):
  def __init__(self,
               protocol="fixbb", 
               use_multimer=False,
               use_templates=False,
               debug=False,
               data_dir=".", 
               **kwargs):  
    assert protocol in ["fixbb","hallucination","binder","partial", "partial_binder"]

    self.protocol = protocol
    self._num = kwargs.pop("num_seq",1)
    self._args = {"use_templates":use_templates, "use_multimer":use_multimer, "use_bfloat16":True,
                  "recycle_mode":"last", "use_mlm": False, "realign": True,
                  "debug":debug, "repeat":False, "homooligomer":False, "copies":1,
                  "optimizer":"sgd", "best_metric":"loss", 
                  "traj_iter":1, "traj_max":10000,
                  "clear_prev": True, "use_dgram":False,
                  "shuffle_first":True, "use_remat":True,
                  "alphabet_size":20, 
                  "use_initial_guess":False, "use_initial_atom_pos":False}

    if self.protocol == "binder": self._args["use_templates"] = True
    if self.protocol == "partial_binder": self._args["use_templates"] = True
    if self.protocol == "partial": self._args["use_templates"] = True
    
    self.opt = {"dropout":True, "pssm_hard":False, "learning_rate":0.1, "norm_seq_grad":True,
                "num_recycles":0, "num_models":1, "sample_models":True,                
                "temp":1.0, "soft":0.0, "hard":0.0, "alpha":2.0,
                "con":      {"num":2, "cutoff":14.0, "binary":False, "seqsep":9, "num_pos":float("inf")},
                "i_con":    {"num":1, "cutoff":21.6875, "binary":False, "num_pos":float("inf")},
                "template": {"rm_ic":False},                
                "weights":  {"seq_ent":0.0, "plddt":0.0, "pae":0.0, "exp_res":0.0, "helix":0.0, "ipsae":0.0, "motif_combined":0.0},
                "rmsd_scale": 3.0,  # for motif_combined loss normalization 1.5
                "fape_cutoff":10.0,
                "crop":False, "crop_indices":[], "crop_target_hotspot_residues":[]} #, "final":False

    self._params = {}
    self._inputs = {}
    self._tmp = {"traj":{"seq":[],"xyz":[],"plddt":[],"pae":[]},
                 "log":[],"best":{}}

    # set arguments/options
    if "initial_guess" in kwargs: kwargs["use_initial_guess"] = kwargs.pop("initial_guess")
    model_names = kwargs.pop("model_names",None)
    keys = list(kwargs.keys())
    for k in keys:
      if k in self._args: self._args[k] = kwargs.pop(k)
      if k in self.opt: self.opt[k] = kwargs.pop(k)

    self.opt['crop'] = bool(self.opt['crop_indices'])

    # collect callbacks
    self._callbacks = {"model": {"pre": kwargs.pop("pre_callback",None),
                                 "post":kwargs.pop("post_callback",None),
                                 "loss":kwargs.pop("loss_callback",None)},
                       "design":{"pre": kwargs.pop("pre_design_callback",None),
                                 "post":kwargs.pop("post_design_callback",None)}}
    
    for m,n in self._callbacks.items():
      for k,v in n.items():
        if v is None: v = []
        if not isinstance(v,list): v = [v]
        self._callbacks[m][k] = v

    if self._args["use_mlm"]:
      self.opt["mlm_dropout"] = 0.15
      self.opt["weights"]["mlm"] = 0.1
    
    assert len(kwargs) == 0, f"ERROR: the following inputs were not set: {kwargs}"

    #############################
    # configure AlphaFold
    #############################
    if self._args["use_multimer"]:
      self._cfg = config.model_config("model_1_multimer")
      # TODO
      self.opt["pssm_hard"] = True
    else:
      self._cfg = config.model_config("model_1_ptm" if self._args["use_templates"] else "model_3_ptm")

    if self._args["recycle_mode"] in ["average","first","last","sample"]:
      num_recycles = 0
    else:
      num_recycles = self.opt["num_recycles"]
    self._cfg.model.num_recycle = num_recycles
    self._cfg.model.global_config.use_remat = self._args["use_remat"]
    self._cfg.model.global_config.use_dgram = self._args["use_dgram"]
    self._cfg.model.global_config.bfloat16  = self._args["use_bfloat16"]

    # load model_params
    if model_names is None:
      model_names = []
      if self._args["use_multimer"]:
        model_names += [f"model_{k}_multimer_v3" for k in [1,2,3,4,5]]
      else:
        if self._args["use_templates"]:
          model_names += [f"model_{k}_ptm" for k in [1,2]]
        else:
          model_names += [f"model_{k}_ptm" for k in [1,2,3,4,5]]

    self._model_params, self._model_names = [],[]
    for model_name in model_names:
      params = data.get_model_haiku_params(model_name=model_name, data_dir=data_dir, fuse=True)
      if params is not None:
        if not self._args["use_multimer"] and not self._args["use_templates"]:
          params = {k:v for k,v in params.items() if "template" not in k}
        self._model_params.append(params)
        self._model_names.append(model_name)
      else:
        print(f"WARNING: '{model_name}' not found")

    #####################################
    # set protocol specific functions
    #####################################
    idx = ["fixbb","hallucination","binder","partial","partial_binder"].index(self.protocol)
    self.prep_inputs = [self._prep_fixbb, self._prep_hallucination, self._prep_binder, self._prep_partial, self._prep_partial_binder][idx]
    self._get_loss   = [self._loss_fixbb, self._loss_hallucination, self._loss_binder, self._loss_partial, self._loss_partial_binder][idx]

  def _get_model(self, cfg, callback=None):
    cfg.model.embeddings_and_evoformer.crop = self.opt["crop"]
    cfg.model.embeddings_and_evoformer.crop_indices = self.opt["crop_indices"]

    a = self._args
    runner = model.RunModel(cfg,
                            recycle_mode=a["recycle_mode"],
                            use_multimer=a["use_multimer"])

    # setup function to get gradients
    def _model(params, model_params, inputs, key):
      inputs["params"] = params
      opt = inputs["opt"]
      to_crop_here = cfg.model.embeddings_and_evoformer.crop
      if cfg.model.embeddings_and_evoformer.crop:
        # Only set hotspot if crop_target_hotspot_residues is not empty
        if self.opt['crop_target_hotspot_residues'] and len(self.opt['crop_target_hotspot_residues']) > 0:
          self.opt['hotspot'] = np.array(self.opt['crop_target_hotspot_residues'], dtype=int)
        elif 'hotspot' in self.opt:
          # Remove hotspot if crop_target_hotspot_residues is empty
          del self.opt['hotspot']
      aux = {}
      key = Key(key=key).get

      #######################################################################
      # INPUTS
      #######################################################################
      # get sequence
      seq = self._get_seq(inputs, aux, key())
      
      # update sequence features      
      pssm = jnp.where(opt["pssm_hard"], seq["hard"], seq["pseudo"])
      if a["use_mlm"]:
        shape = seq["pseudo"].shape[:2]
        mlm = jax.random.bernoulli(key(),opt["mlm_dropout"],shape)
        update_seq(seq["pseudo"], inputs, seq_pssm=pssm, mlm=mlm)
      else:
        if inputs['msa_feat'].shape[1] > seq["pseudo"].shape[1]:
          crop_indices = cfg.model.embeddings_and_evoformer.crop_indices
          to_concat = []
          for i in range(0, len(crop_indices) - 1, 2):
            to_concat.append(jnp.arange(crop_indices[i], crop_indices[i+1]))
          seq_len = len(jnp.concatenate(to_concat, axis=0))
          to_concat.append(jnp.arange(crop_indices[-1], inputs['msa_feat'].shape[1]))
          jax_array_slice = jnp.concatenate(to_concat, axis=0)
          inputs = crop_sizes(inputs, inputs['msa_feat'].shape[1], jax_array_slice)
          to_crop_here = False
          seq = self._get_seq(inputs, aux, key(), seq_len=seq_len) #get new seq with cropped inputs
          pssm = jnp.where(opt["pssm_hard"], seq["hard"], seq["pseudo"])
          self._cfg.model.embeddings_and_evoformer.crop = False
          self.opt["crop"] = False
        update_seq(seq["pseudo"], inputs, seq_pssm=pssm)
      
      # update amino acid sidechain identity
      update_aatype(seq["pseudo"][0].argmax(-1), inputs) 

      # define masks
      inputs["msa_mask"] = jnp.where(inputs["seq_mask"],inputs["msa_mask"],0)

      # update template features
      inputs["mask_template_interchain"] = opt["template"]["rm_ic"]
      if a["use_templates"]:
        if self._cfg.model.embeddings_and_evoformer.crop and to_crop_here:
          inputs['rm_template'].at[:self._target_len].set(False) # #494
          inputs['rm_template_seq'].at[:self._target_len].set(False) #494
          inputs['rm_template_sc'].at[:self._target_len].set(False) #494
        elif self._cfg.model.embeddings_and_evoformer.crop_indices!=[]:
          crop_indices = cfg.model.embeddings_and_evoformer.crop_indices
          to_concat = []
          for i in range(0, len(crop_indices) - 1, 2):
            to_concat.append(jnp.arange(crop_indices[i], crop_indices[i+1]))
          seq_len = len(jnp.concatenate(to_concat, axis=0))
          inputs['rm_template'].at[:seq_len].set(False) # #494
          inputs['rm_template_seq'].at[:seq_len].set(False) #494
          inputs['rm_template_sc'].at[:seq_len].set(False) #494
        self._update_template(inputs, key())
      
      # set dropout
      inputs["use_dropout"] = opt["dropout"]

      if "batch" not in inputs:
        inputs["batch"] = None

      # pre callback
      for fn in self._callbacks["model"]["pre"]:
        fn_args = {"inputs":inputs, "opt":opt, "aux":aux,
                   "seq":seq, "key":key(), "params":params}
        sub_args = {k:fn_args.get(k,None) for k in signature(fn).parameters}
        fn(**sub_args)

      
      #######################################################################
      # OUTPUTS
      #######################################################################

      outputs = runner.apply(model_params, key(), inputs)

      if self._cfg.model.embeddings_and_evoformer.crop and to_crop_here:

        crop_indices = cfg.model.embeddings_and_evoformer.crop_indices
        to_concat = []
        for i in range(0, len(crop_indices) - 1, 2):
          to_concat.append(jnp.arange(crop_indices[i], crop_indices[i+1]))
        seq_len = len(jnp.concatenate(to_concat, axis=0))
        to_concat.append(jnp.arange(crop_indices[-1], inputs['seq_mask'].shape[0]))
        array_slice = jnp.concatenate(to_concat, axis=0)

        inputs = crop_sizes(inputs, inputs['seq_mask'].shape[0], array_slice)

        self._target_len = seq_len
        self._lengths = [seq_len, self._lengths[1]]

        # Only set hotspot if crop_target_hotspot_residues is not empty
        if self.opt['crop_target_hotspot_residues'] and len(self.opt['crop_target_hotspot_residues']) > 0:
          self.opt['hotspot'] = np.array(self.opt['crop_target_hotspot_residues'], dtype=int)
        elif 'hotspot' in self.opt:
          # Remove hotspot if crop_target_hotspot_residues is empty
          del self.opt['hotspot']

      aux.update({"atom_positions": outputs["structure_module"]["final_atom_positions"],
                  "atom_mask":      outputs["structure_module"]["final_atom_mask"],
                  "residue_index":  inputs["residue_index"],
                  "aatype":         inputs["aatype"],
                  "plddt":          get_plddt(outputs),
                  "pae":            get_pae(outputs),
                  "ptm":            get_ptm(inputs, outputs, crop_indices=self._cfg.model.embeddings_and_evoformer.crop_indices),
                  "i_ptm":          get_ptm(inputs, outputs, interface=True, crop_indices=self._cfg.model.embeddings_and_evoformer.crop_indices), 
                  "cmap":           get_contact_map(outputs, opt["con"]["cutoff"]),
                  "i_cmap":         get_contact_map(outputs, opt["i_con"]["cutoff"]),
                  "ipsae":          get_ipsae(outputs, use_contact_map=True, contact_cutoff=opt["i_con"]["cutoff"]),
                  "prev":           outputs["prev"],
                  })

      #######################################################################
      # LOSS
      #######################################################################
      aux["losses"] = {}

      # add protocol specific losses
      if inputs['seq_mask'].shape[0] == outputs['structure_module']['final_atom_positions'].shape[0]:
        self._get_loss(inputs=inputs, outputs=outputs, aux=aux)

      # experimental masked-language-modeling
      if a["use_mlm"]:
        aux["mlm"] = outputs["masked_msa"]["logits"]
        mask = jnp.where(inputs["seq_mask"],mlm,0)
        aux["losses"].update(get_mlm_loss(outputs, mask=mask, truth=seq["pssm"]))

      # run user defined callbacks
      for c in ["loss","post"]:
        for fn in self._callbacks["model"][c]:
          fn_args = {"inputs":inputs, "outputs":outputs, "opt":opt,
                     "aux":aux, "seq":seq, "key":key(), "params":params}
          sub_args = {k:fn_args.get(k,None) for k in signature(fn).parameters}
          if c == "loss": aux["losses"].update(fn(**sub_args))
          if c == "post": fn(**sub_args)

      # save for debugging
      if a["debug"]: aux["debug"] = {"inputs":inputs,"outputs":outputs}
  
      # weighted loss
      w = opt["weights"]
      loss = sum([v * w[k] if k in w else v for k,v in aux["losses"].items()])

      return loss, aux
    
    return {"grad_fn":jax.jit(jax.value_and_grad(_model, has_aux=True, argnums=0)),
            "fn":jax.jit(_model), "runner":runner}
