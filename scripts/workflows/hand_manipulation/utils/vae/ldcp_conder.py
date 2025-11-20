# adpat from https://github.com/ldcq/ldcq/blob/7ec96f3682e04e89385fe18e17a20f5315f0048d/models/skill_model.py
import numpy as np
import torch
import torch.nn as nn

import torch.distributions.kl as KL
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder


class AbstractDynamics(BaseDecoder):
    '''
    P(s_T|s_0,z) is our "abstract dynamics model", because it predicts the resulting state transition over T timesteps given a skill 
    (so similar to regular dynamics model, but in skill space and also temporally extended)
    See Encoder and Decoder for more description
    '''

    def __init__(self,
                 state_dim,
                 latent_dim,
                 layer_dims,
                 per_element_sigma=True):

        super(AbstractDynamics, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim + latent_dim, layer_dims), nn.ReLU(),
            nn.Linear(layer_dims, layer_dims), nn.ReLU())
        self.mean_layer = nn.Sequential(nn.Linear(layer_dims, layer_dims),
                                        nn.ReLU(),
                                        nn.Linear(layer_dims, state_dim))
        if per_element_sigma:
            self.sig_layer = nn.Sequential(nn.Linear(layer_dims, layer_dims),
                                           nn.ReLU(),
                                           nn.Linear(layer_dims, state_dim),
                                           nn.Softplus())
        else:
            self.sig_layer = nn.Sequential(nn.Linear(layer_dims, layer_dims),
                                           nn.ReLU(), nn.Linear(layer_dims, 1),
                                           nn.Softplus())

        self.state_dim = state_dim
        self.per_element_sigma = per_element_sigma

    def forward(self, s0, z):
        '''
        INPUTS:
            s0: batch_size x 1 x state_dim initial state (first state in execution of skill)
            z:  batch_size x 1 x latent_dim "skill"/z
        OUTPUTS: 
            sT_mean: batch_size x 1 x state_dim tensor of terminal (time=T) state means
            sT_sig:  batch_size x 1 x state_dim tensor of terminal (time=T) state standard devs
        '''

        # concatenate s0 and z
        s0_z = torch.cat([s0, z], dim=-1)
        # pass s0_z through layers
        feats = self.layers(s0_z)
        # get mean and stand dev of action distribution
        sT_mean = self.mean_layer(feats)
        sT_sig = self.sig_layer(feats)

        if not self.per_element_sigma:
            sT_sig = torch.cat(self.state_dim * [sT_sig], dim=-1)

        return sT_mean, sT_sig


class AutoregressiveStateDecoder(BaseDecoder):
    '''
    P(s_T|s_0,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    See Encoder and Decoder for more description
    '''

    def __init__(self,
                 state_dim,
                 latent_dim,
                 layer_dims,
                 per_element_sigma=True):

        super(AutoregressiveStateDecoder, self).__init__()
        self.decoder_components = nn.ModuleList([
            LowLevelPolicy(state_dim + i,
                           1,
                           latent_dim,
                           layer_dims,
                           a_dist='normal') for i in range(state_dim)
        ])
        self.state_dim = state_dim

    def forward(self, state, s_T, z, evaluation=False):
        '''
        INPUTS:
            state: batch_size x 1 x state_dim tensor of states 
            action: batch_size x 1 x action_dim tensor of actions
            z:     batch_size x 1 x latent_dim tensor of states
        OUTPUTS:
            a_mean: batch_size x T x action_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x T x action_dim tensor of action standard devs for each t in {0.,,,.T}
        
        Iterate through each low level policy component.
        The ith element gets to condition on all elements up to but NOT including a_i
        '''
        s_means = []
        s_sigs = []

        s_means_tensor = torch.zeros_like(state)
        s_sigs_tensor = torch.zeros_like(state)

        for i in range(self.state_dim):
            # Concat state, and a up to i.  state_a takes place of state in orginary policy.
            if not evaluation:
                state_a = torch.cat([state, s_T[:, :, :i]], dim=-1)
            else:
                state_a = torch.cat([state, s_means_tensor[:, :, :i].detach()],
                                    dim=-1)
            # pass through ith policy component
            s_T_mean_i, s_T_sig_i = self.decoder_components[i](
                state_a, z)  # these are batch_size x T x 1
            # add to growing list of policy elements
            s_means.append(s_T_mean_i)
            s_sigs.append(s_T_sig_i)

            if evaluation:
                s_means_tensor = torch.cat(s_means, dim=-1)
                s_sigs_tensor = torch.cat(s_sigs, dim=-1)

        s_means = torch.cat(s_means, dim=-1)
        s_sigs = torch.cat(s_sigs, dim=-1)
        return s_means, s_sigs

    def sample(self, state, z):
        states = []
        for i in range(self.state_dim):
            # Concat state, a up to i, and z_tiled
            state_a = torch.cat([state] + states, dim=-1)
            # pass through ith policy component
            s_T_mean_i, s_T_sig_i = self.decoder_components[i](
                state_a, z)  # these are batch_size x T x 1
            s_i = reparameterize(s_T_mean_i, s_T_sig_i)
            states.append(s_i)

        return torch.cat(states, dim=-1)

    def numpy_dynamics(self, state, z):
        '''
        maps state as a numpy array and z as a pytorch tensor to a numpy action
        '''
        state = torch.reshape(
            torch.tensor(state,
                         device=torch.device('cuda:0'),
                         dtype=torch.float32), (1, 1, -1))

        s_T = self.sample(state, z)
        s_T = s_T.detach().cpu().numpy()

        return s_T.reshape([
            self.state_dim,
        ])


class LowLevelPolicy(BaseDecoder):
    '''
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    See Encoder and Decoder for more description
    '''

    def __init__(self,
                 state_dim,
                 action_dim,
                 latent_dim,
                 layer_dims,
                 a_dist,
                 fixed_sig=None):

        super(LowLevelPolicy, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim + latent_dim, layer_dims), nn.ReLU(),
            nn.Linear(layer_dims, layer_dims), nn.ReLU())
        if a_dist == 'softmax':
            self.mean_layer = nn.Sequential(
                nn.Linear(layer_dims, layer_dims), nn.ReLU(),
                nn.Linear(layer_dims,
                          21))  #ONLY FOR AUTOREGRESSIVE POLICY DECODER
            self.act = nn.Softmax(dim=2)
        else:
            self.mean_layer = nn.Sequential(nn.Linear(layer_dims, layer_dims),
                                            nn.ReLU(),
                                            nn.Linear(layer_dims, action_dim))
            self.sig_layer = nn.Sequential(nn.Linear(layer_dims, layer_dims),
                                           nn.ReLU(),
                                           nn.Linear(layer_dims, action_dim))
        self.a_dist = a_dist
        self.action_dim = action_dim
        self.fixed_sig = fixed_sig

    def forward(self, state, z):
        '''
        INPUTS:
            state: batch_size x T x state_dim tensor of states 
            z:     batch_size x 1 x latent_dim tensor of states
        OUTPUTS:
            a_mean: batch_size x T x action_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x T x action_dim tensor of action standard devs for each t in {0.,,,.T}
        '''
        # tile z along time axis so dimension matches state
        z_tiled = z.tile([1, state.shape[-2], 1])  #not sure about this

        # Concat state and z_tiled
        state_z = torch.cat([state, z_tiled], dim=-1)
        # pass z and state through layers
        feats = self.layers(state_z)
        # get mean and stand dev of action distribution
        a_mean = self.mean_layer(feats)
        if self.a_dist == 'softmax':
            a_mean = self.act(a_mean)
            return a_mean, None
        a_sig = nn.Softplus()(self.sig_layer(feats))

        if self.fixed_sig is not None:
            a_sig = self.fixed_sig * torch.ones_like(a_sig)

        return a_mean, a_sig

    def numpy_policy(self, state, z):
        '''
        maps state as a numpy array and z as a pytorch tensor to a numpy action
        '''
        state = torch.reshape(
            torch.tensor(state,
                         device=torch.device('cuda:0'),
                         dtype=torch.float32), (1, 1, -1))

        a_mean, a_sig = self.forward(state, z)
        action = self.reparameterize(a_mean, a_sig)
        if self.a_dist == 'tanh_normal':
            action = nn.Tanh()(action)
        action = action.detach().cpu().numpy()

        return action.reshape([
            self.action_dim,
        ])

    def reparameterize(self, mean, std):
        if self.a_dist == 'softmax':
            intervals = torch.linspace(-1, 1, 21).cuda()
            max_idx = torch.argmax(mean, dim=2).unsqueeze(2)
            max_interval = intervals[max_idx]
            return max_interval
        eps = torch.normal(
            torch.zeros(mean.size()).cuda(),
            torch.ones(mean.size()).cuda())
        return mean + std * eps


class AutoregressiveLowLevelPolicy(BaseDecoder):
    '''
    P(a_t|s_t,z) is our "low-level policy", so this is the feedback policy the agent runs while executing skill described by z.
    See Encoder and Decoder for more description
    '''

    def __init__(self,
                 state_dim,
                 action_dim,
                 latent_dim,
                 layer_dims,
                 chunk_size,
                 device='cuda',
                 a_dist='normal',
                 fixed_sig=None):

        super(AutoregressiveLowLevelPolicy, self).__init__()
        self.policy_components = nn.ModuleList([
            LowLevelPolicy(state_dim + i,
                           1,
                           latent_dim,
                           layer_dims,
                           a_dist=a_dist,
                           fixed_sig=fixed_sig) for i in range(action_dim)
        ]).to(device)
        self.action_dim = action_dim
        self.a_dist = a_dist

    def forward(self, data):
        '''
        INPUTS:
            state: batch_size x T x state_dim tensor of states
            action: batch_size x T x action_dim tensor of actions
            z:     batch_size x 1 x latent_dim tensor of states
        OUTPUTS:
            a_mean: batch_size x T x action_dim tensor of action means for each t in {0.,,,.T}
            a_sig:  batch_size x T x action_dim tensor of action standard devs for each t in {0.,,,.T}
        
        Iterate through each low level policy component.
        The ith element gets to condition on all elements up to but NOT including a_i
        '''

        state = data["state"]
        actions = data["action_chunk"]
        z = data["z"]
        a_means = []
        a_sigs = []
        for i in range(self.action_dim):
            # Concat state, and a up to i.  state_a takes place of state in orginary policy.
            state_a = torch.cat([state, actions[:, :, :i]], dim=-1)
            # pass through ith policy component
            a_mean_i, a_sig_i = self.policy_components[i](
                state_a, z)  # these are batch_size x T x 1
            if self.a_dist == 'softmax':
                a_mean_i = a_mean_i.unsqueeze(dim=2)
            # add to growing list of policy elements
            a_means.append(a_mean_i)
            if not self.a_dist == 'softmax':
                a_sigs.append(a_sig_i)
        if self.a_dist == 'softmax':
            a_means = torch.cat(a_means, dim=2)
            return a_means, None
        a_means = torch.cat(a_means, dim=-1)
        a_sigs = torch.cat(a_sigs, dim=-1)
        import pdb
        pdb.set_trace()

        return ModelOutput(reconstruction=a_sigs)

    def sample(self, state, z):
        actions = []
        for i in range(self.action_dim):
            # Concat state, a up to i, and z_tiled
            state_a = torch.cat([state] + actions, dim=-1)
            # pass through ith policy component
            a_mean_i, a_sig_i = self.policy_components[i](
                state_a, z)  # these are batch_size x T x 1

            a_i = self.reparameterize(a_mean_i, a_sig_i)
            #a_i = a_mean_i

            if self.a_dist == 'tanh_normal':
                a_i = nn.Tanh()(a_i)
            actions.append(a_i)

        return torch.cat(actions, dim=-1)

    def numpy_policy(self, state, z):
        '''
        maps state as a numpy array and z as a pytorch tensor to a numpy action
        '''
        state = torch.reshape(
            torch.tensor(state,
                         device=torch.device('cuda:0'),
                         dtype=torch.float32), (1, 1, -1))

        action = self.sample(state, z)
        action = action.detach().cpu().numpy()

        return action.reshape([
            self.action_dim,
        ])

    def reparameterize(self, mean, std):
        if self.a_dist == 'softmax':
            intervals = torch.linspace(-1, 1, 21).cuda()
            # max_idx = torch.distributions.categorical.Categorical(mean).sample()
            max_idx = torch.argmax(mean, dim=2)
            max_interval = intervals[max_idx]
            return max_interval.unsqueeze(-1)
        eps = torch.normal(
            torch.zeros(mean.size()).cuda(),
            torch.ones(mean.size()).cuda())
        return mean + std * eps


class SimpleLowLevelPolicy(BaseDecoder):

    def __init__(self,
                 state_dim,
                 action_dim,
                 latent_dim,
                 layer_dims,
                 action_chunk_len=10,
                 device='cuda',
                 hidden_dim=128,
                 use_state=True):
        super().__init__()
        if use_state:
            input_dim = state_dim + latent_dim
        else:
            input_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # Output full chunk: (T, action_dim)
            nn.Linear(hidden_dim, action_chunk_len * action_dim)).to(device)
        self.use_state = use_state

        self.action_chunk_len = action_chunk_len
        self.action_dim = action_dim

    def forward(self, data):
        '''
        Inputs:
            s0: (B, state_dim) — the initial state
            z:  (B, latent_dim) — the latent skill code
        Output:
            reconstruction: (B, T, action_dim)
        '''

        z = data["z"]  # (B, 1, latent_dim)
        s0 = data["state"][:, 0, :]  # (B, state_dim)

        if self.use_state:

            x = torch.cat([s0, z.reshape(z.shape[0], -1)],
                          dim=-1)  # (B, state+z)
        else:

            x = z.reshape(z.shape[0], -1)
        out = self.net(x)  # (B, T * action_dim)
        out = out.view(-1, self.action_chunk_len, self.action_dim)
        return ModelOutput(reconstruction=out)


class LDCPTransformEncoder(BaseEncoder):
    '''
    Encoder module.
    -Concat states+actions
    -Pass through linear embedding
    -Pass through bidirectional RNN
    -Pass output of bidirectional RNN through 2 linear layers, one to get mean of z and one to get stand dev (we're estimating one z ("skill") for entire episode)
    '''

    def __init__(self,
                 state_dim,
                 action_dim,
                 latent_dim,
                 layer_dims,
                 chunk_size,
                 device='cuda',
                 n_layers=3,
                 n_heads=4,
                 dropout=0.1,
                 use_vae=False,
                 use_state=True):
        super(LDCPTransformEncoder, self).__init__()
        self.device = device

        self.chunk_size = chunk_size
        self.embed_state = torch.nn.Linear(state_dim, layer_dims).to(device)
        self.embed_action = torch.nn.Linear(action_dim, layer_dims).to(device)
        self.embed_ln = nn.LayerNorm(layer_dims).to(device)

        # Last token is special -> used for z prediction
        self.embed_timestep = nn.Embedding(chunk_size + 1,
                                           layer_dims).to(device)

        encoder_layer = nn.TransformerEncoderLayer(layer_dims,
                                                   nhead=n_heads,
                                                   dim_feedforward=4 *
                                                   layer_dims,
                                                   dropout=dropout).to(device)
        self.transformer_model = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers).to(device)

        self.mean_layer = nn.Sequential(nn.Linear(layer_dims, layer_dims),
                                        nn.ReLU(),
                                        nn.Linear(layer_dims,
                                                  latent_dim)).to(device)
        self.use_vae = use_vae
        if use_vae:

            self.sig_layer = nn.Sequential(nn.Linear(layer_dims, layer_dims),
                                           nn.ReLU(),
                                           nn.Linear(layer_dims, latent_dim),
                                           nn.Softplus()).to(device)

    def forward(self, data):
        '''
        Takes a sequence of states and actions, and infers the distribution over latent skill variable, z
        
        INPUTS:
            states: batch_size x T x state_dim state sequence tensor
            actions: batch_size x T x action_dim action sequence tensor
        OUTPUTS:
            z_mean: batch_size x 1 x latent_dim tensor indicating mean of z distribution
            z_sig:  batch_size x 1 x latent_dim tensor indicating standard deviation of z distribution
        '''

        actions = data["action_chunk"]
        states = data["state"]
        timesteps = self.embed_timestep(
            torch.arange(actions.shape[1]).to(actions.device))
        timesteps = timesteps.unsqueeze(0).repeat((actions.shape[0], 1, 1))

        z_embedding = self.embed_timestep(
            torch.LongTensor([self.chunk_size]).to(actions.device))
        z_embedding = z_embedding.unsqueeze(0).repeat((actions.shape[0], 1, 1))

        state_latent = self.embed_state(states) + timesteps
        action_latent = self.embed_action(actions) + timesteps

        transformer_inputs = torch.cat(
            [state_latent, action_latent, z_embedding], dim=1)
        transformer_inputs = self.embed_ln(transformer_inputs)

        transformer_outputs = self.transformer_model(transformer_inputs)

        hn = transformer_outputs[:, -1]

        z_mean = self.mean_layer(hn).unsqueeze(1)
        if self.use_vae:
            z_sig = self.sig_layer(hn).unsqueeze(1)

            return ModelOutput(embedding=z_mean, log_covariance=z_sig)
        else:
            return ModelOutput(embedding=z_mean)


class SimpleActionStateEncoder(BaseEncoder):

    def __init__(self,
                 state_dim,
                 action_dim,
                 latent_dim,
                 layer_dims,
                 chunk_size,
                 device='cuda',
                 hidden_dims=[16],
                 use_vae=False,
                 use_state=True):
        super().__init__()
        self.device = device
        layers = []
        self.latent_dim = latent_dim
        dims = [state_dim + action_dim] + hidden_dims + [latent_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers).to(device)

    def forward(self, data):

        actions = data["action_chunk"]
        states = data["state"]
        self.embedding = self.model(torch.cat([actions, states], dim=-1))

        return ModelOutput(embedding=self.embedding)


class SimpleStateActionDecoder(BaseDecoder):

    def __init__(
        self,
        state_dim,
        action_dim,
        latent_dim,
        layer_dims,
        action_chunk_len=10,
        device='cuda',
        hidden_dim=128,
        use_state=True,
        hidden_dims=[16],
        use_vae=False,
    ):
        super().__init__()
        # Decoder MLP
        layers = []
        dims = [latent_dim + state_dim] + hidden_dims + [action_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.decoder = nn.Sequential(*layers).to(device)

    def forward(self, data):
        z = data["z"]  # (B, 1, latent_dim)
        s0 = data["state"][:, 0, :]  # (B, state_dim)

        z = torch.cat([z.reshape(z.shape[0], -1), s0], dim=-1)  # (B, state+z)

        return ModelOutput(reconstruction=self.decoder(z))


class LDCPGRUEncoder(BaseEncoder):
    '''
    Encoder module.
    -Concat states+actions
    -Pass through linear embedding
    -Pass through bidirectional RNN
    -Pass output of bidirectional RNN through 2 linear layers, one to get mean of z and one to get stand dev (we're estimating one z ("skill") for entire episode)
    '''

    def __init__(self,
                 state_dim,
                 action_dim,
                 latent_dim,
                 layer_dims,
                 chunk_size,
                 device='cuda',
                 n_gru_layers=4,
                 normalize_latent=False,
                 use_vae=False,
                 use_state=True):
        super(LDCPGRUEncoder, self).__init__()
        self.use_state = use_state
        self.device = device

        self.state_dim = state_dim  # state dimension
        self.action_dim = action_dim  # action dimension
        self.normalize_latent = normalize_latent
        self.latent_dim = latent_dim  # latent dimension

        self.emb_layer = nn.Sequential(nn.Linear(state_dim, layer_dims),
                                       nn.ReLU(),
                                       nn.Linear(layer_dims, layer_dims),
                                       nn.ReLU()).to(device)
        if self.use_state:
            self.rnn = nn.GRU(layer_dims + action_dim,
                              layer_dims,
                              batch_first=True,
                              bidirectional=True,
                              num_layers=n_gru_layers).to(device)
        else:

            self.rnn = nn.GRU(action_dim,
                              layer_dims,
                              batch_first=True,
                              bidirectional=True,
                              num_layers=n_gru_layers).to(device)
        #self.mean_layer = nn.Linear(layer_dims,latent_dim)
        self.mean_layer = nn.Sequential(nn.Linear(2 * layer_dims, layer_dims),
                                        nn.ReLU(),
                                        nn.Linear(layer_dims,
                                                  latent_dim)).to(device)
        #self.sig_layer  = nn.Sequential(nn.Linear(layer_dims,latent_dim),nn.Softplus())  # using softplus to ensure stand dev is positive
        self.use_vae = use_vae
        if use_vae:
            self.sig_layer = nn.Sequential(
                nn.Linear(2 * layer_dims, layer_dims), nn.ReLU(),
                nn.Linear(layer_dims, latent_dim), nn.Softplus()).to(device)

    def forward(self, data):
        '''
        Takes a sequence of states and actions, and infers the distribution over latent skill variable, z
        
        INPUTS:
            states: batch_size x T x state_dim state sequence tensor
            actions: batch_size x T x action_dim action sequence tensor
        OUTPUTS:
            z_mean: batch_size x 1 x latent_dim tensor indicating mean of z distribution
            z_sig:  batch_size x 1 x latent_dim tensor indicating standard deviation of z distribution
        '''
        actions = data["action_chunk"]
        states = data["state"]

        s_emb = self.emb_layer(states)
        # through rnn
        if self.use_state:
            s_emb_a = torch.cat([s_emb, actions], dim=-1)
        else:
            s_emb_a = actions

        feats, _ = self.rnn(s_emb_a)
        hn = feats[:, -1:, :]

        z_mean = self.mean_layer(hn).reshape(
            actions.shape[0],
            -1,
        )

        if self.normalize_latent:
            z_mean = z_mean / torch.norm(z_mean, dim=-1).unsqueeze(-1)
        if self.use_vae:
            z_sig = self.sig_layer(hn)
            return ModelOutput(embedding=z_mean, log_covariance=z_sig)
        else:
            return ModelOutput(embedding=z_mean)


# adapted from reactive difffusion policy : https://github.com/xiaoxiaoxh/reactive_diffusion_policy/blob/main/reactive_diffusion_policy/model/vae/model.py
import einops


def weights_init_encoder(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class ReactiveDiffEncoderCNN(BaseEncoder):

    def __init__(
        self,
        state_dim,
        action_dim,
        latent_dim,
        n_conv_layers=4,
        hidden_dim=128,
        device='cuda',
        normalize_latent=False,
        use_vae=False,
        use_state=True,
    ):
        super(ReactiveDiffEncoderCNN, self).__init__()

        self.action_dim = action_dim
        self.device = device

        # Build the convolutional encoder
        layers = []
        in_channels = action_dim
        for _ in range(n_conv_layers):
            layers.append(
                nn.Conv1d(in_channels,
                          hidden_dim,
                          kernel_size=5,
                          stride=2,
                          padding=2))
            layers.append(nn.ReLU())
            in_channels = hidden_dim  # update for next layer

        # Final projection to latent space
        layers.append(
            nn.Conv1d(in_channels,
                      latent_dim,
                      kernel_size=5,
                      stride=2,
                      padding=2))

        self.encoder = nn.Sequential(*layers)
        self.apply(weights_init_encoder)

    def forward(self, data, flatten=False):
        """
        Args:
            data: Dict with keys:
                - action_chunk: (N, T, A)
                - state: (N, ...)
            flatten: If True, flatten the output to (N, T*C)

        Returns:
            Tensor of shape (N, T', C) or (N, T'*C) if flatten=True
        """
        actions = data["action_chunk"]  # shape (N, T, A)
        x = einops.rearrange(actions,
                             "N T A -> N A T")  # for Conv1d: (N, C, L)

        h = self.encoder(x)  # shape: (N, latent_dim, T')

        h = einops.rearrange(h, "N C T -> N T C")

        if flatten:
            h = einops.rearrange(h, "N T C -> N (T C)")

        return ModelOutput(embedding=h.reshape(
            actions.shape[0],
            -1,
        ))


class ReactiveDiffDecoderRNN(BaseDecoder):

    def __init__(self,
                 state_dim,
                 action_dim,
                 latent_dim,
                 layer_dims,
                 action_chunk_len=10,
                 device='cuda',
                 hidden_dim=128,
                 use_state=True):
        super(ReactiveDiffDecoderRNN, self).__init__()
        self.rnn = nn.GRU(latent_dim + state_dim,
                          hidden_dim,
                          layer_dims,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim * action_chunk_len)
        self.action_dim = action_dim
        self.apply(weights_init_encoder)

    def forward(self, data):

        x = torch.cat([data["z"][:, None], data["state"][:, 0][:, None]],
                      dim=-1)
        x, _ = self.rnn(x)
        x = self.fc(x)

        x = einops.rearrange(x, "N T A -> N (T A)").reshape(
            x.shape[0], -1, self.action_dim)

        return ModelOutput(reconstruction=x)
