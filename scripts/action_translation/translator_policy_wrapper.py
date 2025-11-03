"""Policy wrapper that applies action translation on top of a source policy.

This wrapper intercepts actions from a source policy and translates them using
a trained action translator model before sending them to the environment.

Example usage:
    # In your evaluation script:
    from scripts.action_translation.translator_policy_wrapper import TranslatorPolicyWrapper

    # Load your source policy as usual
    source_policy = runner.get_inference_policy(device=device)

    # Wrap it with the action translator
    translator_config = "configs/action_translator/kuka_allegro_mlp.yaml"
    translator_checkpoint = "trained_models/action_translator/normal_to_slippery/model.pth"
    policy = TranslatorPolicyWrapper(
        source_policy=source_policy,
        translator_config=translator_config,
        translator_checkpoint=translator_checkpoint,
        device=device
    )

    # Use policy as normal - it will automatically translate actions
    actions = policy(obs)
"""

import os
import sys
import yaml
import torch
import inspect
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append('/home/raymond/projects/generative-policies')


class TranslatorPolicyWrapper:
    """
    Wrapper that applies action translation to a source policy.

    This class wraps a source policy and an action translator model,
    automatically translating actions from the source domain to the target domain.
    """

    def __init__(
        self,
        source_policy,
        translator_config: str,
        translator_checkpoint: str,
        env=None,
        device: str = 'cuda',
        verbose: bool = True
    ):
        """
        Initialize the translator policy wrapper.

        Args:
            source_policy: The source domain policy (e.g., normal friction policy)
            translator_config: Path to action translator config YAML file
            translator_checkpoint: Path to trained action translator checkpoint (.pth)
            env: Isaac Lab environment (needed for extracting 134D states)
            device: Device to run the translator on
            verbose: Whether to print loading information
        """
        self.source_policy = source_policy
        self.env = env
        self.device = device
        self.verbose = verbose

        # Load action translator
        self.translator, self.checkpoint = self._load_translator(
            translator_config, translator_checkpoint
        )

        # Import state extractor
        from scripts.lib.state_extraction import extract_from_env
        self.extract_from_env = extract_from_env

        if verbose:
            print("=" * 80)
            print("TRANSLATOR POLICY WRAPPER INITIALIZED")
            print("=" * 80)
            print(f"Translator checkpoint: {translator_checkpoint}")
            print(f"Normalization enabled: {self.checkpoint.get('normalize_data', False)}")
            print(f"Environment provided: {env is not None}")
            print("=" * 80 + "\n")

    def _load_translator(self, config_path: str, checkpoint_path: str):
        """Load action translator model from config and checkpoint."""
        if self.verbose:
            print(f"Loading action translator from {checkpoint_path}")

        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        model_config = config.get('model', {})

        # Build model from config
        target_path = model_config['_target_']
        module_path, class_name = target_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        model_class = getattr(module, class_name)

        # Get init parameters
        init_params = set(inspect.signature(model_class.__init__).parameters.keys())
        init_params.discard('self')

        # Build model parameters (dimensions will be overridden from checkpoint)
        model_params = {k: v for k, v in model_config.items() if k in init_params}
        model_params['device'] = self.device

        # Create model
        model = model_class(**model_params)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        if self.verbose:
            print(f"Action translator loaded successfully")

        return model, checkpoint

    def _normalize_inputs(self, states, actions_src):
        """Normalize states and source actions using checkpoint statistics."""
        normalize_data = self.checkpoint.get('normalize_data', False)

        if not normalize_data:
            return states, actions_src

        # Normalize states
        # Stats are stored as tuples: 'states': (mean_tensor, std_tensor)
        state_stats = self.checkpoint.get('states')
        if state_stats is not None:
            state_mean, state_std = state_stats
            state_mean = state_mean.to(self.device)
            state_std = state_std.to(self.device)

            # Handle case where stats have extra dimension (shape [1, D] instead of [D])
            if state_mean.dim() > 1:
                state_mean = state_mean.squeeze(0)
            if state_std.dim() > 1:
                state_std = state_std.squeeze(0)

            states = (states - state_mean) / state_std

        # Normalize source actions
        action_src_stats = self.checkpoint.get('actions_src')
        if action_src_stats is not None:
            action_src_mean, action_src_std = action_src_stats
            action_src_mean = action_src_mean.to(self.device)
            action_src_std = action_src_std.to(self.device)

            # Handle case where stats have extra dimension (shape [1, D] instead of [D])
            if action_src_mean.dim() > 1:
                action_src_mean = action_src_mean.squeeze(0)
            if action_src_std.dim() > 1:
                action_src_std = action_src_std.squeeze(0)

            actions_src = (actions_src - action_src_mean) / action_src_std

        return states, actions_src

    def _denormalize_outputs(self, actions_target):
        """Denormalize target actions using checkpoint statistics."""
        normalize_data = self.checkpoint.get('normalize_data', False)

        if not normalize_data:
            return actions_target

        # Denormalize target actions
        # Stats are stored as tuples: 'actions_target': (mean_tensor, std_tensor)
        action_target_stats = self.checkpoint.get('actions_target')
        if action_target_stats is not None:
            action_target_mean, action_target_std = action_target_stats
            action_target_mean = action_target_mean.to(self.device)
            action_target_std = action_target_std.to(self.device)

            # Handle case where stats have extra dimension (shape [1, D] instead of [D])
            if action_target_mean.dim() > 1:
                action_target_mean = action_target_mean.squeeze(0)
            if action_target_std.dim() > 1:
                action_target_std = action_target_std.squeeze(0)

            actions_target = actions_target * action_target_std + action_target_mean

        return actions_target

    def __call__(self, obs):
        """
        Get translated actions for the given observations.

        Args:
            obs: Observations (can be dict/TensorDict or tensor)

        Returns:
            Translated actions (same format as source policy output)
        """
        # Step 1: Get source policy actions
        with torch.no_grad():
            actions_src = self.source_policy(obs)

        # Step 2: Extract 134D state from environment
        # This matches the state representation used during translator training
        # Pass None for obs_buf to use env.obs_buf directly (not the wrapped obs)
        states = self.extract_from_env(self.env, obs_buf=None)

        # Ensure tensors are on correct device
        states = states.to(self.device)
        actions_src = actions_src.to(self.device)

        # Step 3: Normalize inputs
        states_norm, actions_src_norm = self._normalize_inputs(states, actions_src)

        # Step 4: Translate actions
        with torch.no_grad():
            actions_target_norm = self.translator.predict(states_norm, actions_src_norm)

            # Convert to tensor if numpy
            if isinstance(actions_target_norm, np.ndarray):
                actions_target_norm = torch.from_numpy(actions_target_norm).to(self.device)

        # Step 5: Denormalize outputs
        actions_target = self._denormalize_outputs(actions_target_norm)

        return actions_target

    def reset(self):
        """Reset policy state (if source policy has reset method)."""
        if hasattr(self.source_policy, 'reset'):
            self.source_policy.reset()


def load_translator_policy(
    source_policy,
    translator_config: str,
    translator_checkpoint: str,
    device: str = 'cuda',
    verbose: bool = True
):
    """
    Convenience function to load a translator policy wrapper.

    Args:
        source_policy: The source domain policy
        translator_config: Path to action translator config YAML file
        translator_checkpoint: Path to trained action translator checkpoint (.pth)
        device: Device to run on
        verbose: Whether to print loading information

    Returns:
        TranslatorPolicyWrapper instance
    """
    return TranslatorPolicyWrapper(
        source_policy=source_policy,
        translator_config=translator_config,
        translator_checkpoint=translator_checkpoint,
        device=device,
        verbose=verbose
    )
