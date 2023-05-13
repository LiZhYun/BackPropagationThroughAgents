from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import numpy as np
from bta.algorithms.utils.gobigger.nn_module import fc_block, conv2d_block, MLP
from bta.algorithms.utils.gobigger.res_block import ResBlock
from bta.algorithms.utils.gobigger.transformer import Transformer

def sequence_mask(lengths: torch.Tensor, max_len: Optional[int] =None):
    r"""
        Overview:
            create a mask for a batch sequences with different lengths
        Arguments:
            - lengths (:obj:`tensor`): lengths in each different sequences, shape could be (n, 1) or (n)
            - max_len (:obj:`int`): the padding size, if max_len is None, the padding size is the
                max length of sequences
        Returns:
            - masks (:obj:`torch.BoolTensor`): mask has the same device as lengths
    """
    if len(lengths.shape) == 1:
        lengths = lengths.unsqueeze(dim=1)
    bz = lengths.numel()
    if max_len is None:
        max_len = lengths.max()
    return torch.arange(0, max_len).type_as(lengths).repeat(bz, 1).lt(lengths).to(lengths.device)

class ScatterConnection(nn.Module):
    r"""
        Overview:
            Scatter feature to its corresponding location
            In alphastar, each entity is embedded into a tensor, these tensors are scattered into a feature map
            with map size
    """

    def __init__(self, scatter_type='add') -> None:
        r"""
            Overview:
                Init class
            Arguments:
                - scatter_type (:obj:`str`): add or cover, if two entities have same location, scatter type decides the
                    first one should be covered or added to second one
        """
        super(ScatterConnection, self).__init__()
        self.scatter_type = scatter_type
        assert self.scatter_type in ['cover', 'add']

    def xy_forward(self, x: torch.Tensor, spatial_size: Tuple[int, int], coord_x: torch.Tensor,coord_y) -> torch.Tensor:
        device = x.device
        BatchSize, Num, EmbeddingSize = x.shape
        x = x.permute(0, 2, 1)
        H, W = spatial_size
        indices = (coord_x * W + coord_y).long()
        indices = indices.unsqueeze(dim=1).repeat(1, EmbeddingSize, 1)
        output = torch.zeros(size=(BatchSize, EmbeddingSize, H, W), device=device).view(BatchSize, EmbeddingSize,
                                                                                         H * W)
        if self.scatter_type == 'cover':
            output.scatter_(dim=2, index=indices, src=x)
        elif self.scatter_type == 'add':
            output.scatter_add_(dim=2, index=indices, src=x)
        output = output.view(BatchSize, EmbeddingSize, H, W)
        return output

    def forward(self, x: torch.Tensor, spatial_size: Tuple[int, int], location: torch.Tensor) -> torch.Tensor:
        """
            Overview:
                scatter x into a spatial feature map
            Arguments:
                - x (:obj:`tensor`): input tensor :math: `(B, M, N)` where `M` means the number of entity, `N` means\
                  the dimension of entity attributes
                - spatial_size (:obj:`tuple`): Tuple[H, W], the size of spatial feature x will be scattered into
                - location (:obj:`tensor`): :math: `(B, M, 2)` torch.LongTensor, each location should be (y, x)
            Returns:
                - output (:obj:`tensor`): :math: `(B, N, H, W)` where `H` and `W` are spatial_size, return the\
                    scattered feature map
            Shapes:
                - Input: :math: `(B, M, N)` where `M` means the number of entity, `N` means\
                  the dimension of entity attributes
                - Size: Tuple[H, W]
                - Location: :math: `(B, M, 2)` torch.LongTensor, each location should be (y, x)
                - Output: :math: `(B, N, H, W)` where `H` and `W` are spatial_size

            .. note::
                when there are some overlapping in locations, ``cover`` mode will result in the loss of information, we
                use the addition as temporal substitute.
        """
        device = x.device
        BatchSize, Num, EmbeddingSize = x.shape
        x = x.permute(0, 2, 1)
        H, W = spatial_size
        indices = location[:, :, 1] + location[:, :, 0] * W
        indices = indices.unsqueeze(dim=1).repeat(1, EmbeddingSize, 1)
        output = torch.zeros(size=(BatchSize, EmbeddingSize, H, W), device=device).view(BatchSize, EmbeddingSize,
                                                                                         H * W)
        if self.scatter_type == 'cover':
            output.scatter_(dim=2, index=indices, src=x)
        elif self.scatter_type == 'add':
            output.scatter_add_(dim=2, index=indices, src=x)
        output = output.view(BatchSize, EmbeddingSize, H, W)

        return output

class OnehotEncoder(nn.Module):
    def __init__(self, num_embeddings: int):
        super(OnehotEncoder, self).__init__()
        self.num_embeddings = num_embeddings
        self.main = nn.Embedding.from_pretrained(torch.eye(self.num_embeddings), freeze=True,
                                                 padding_idx=None)

    def forward(self, x: torch.Tensor):
        x = x.long().clamp_(max=self.num_embeddings - 1)
        return self.main(x)


class OnehotEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(OnehotEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.main = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim)

    def forward(self, x: torch.Tensor):
        x = x.long().clamp_(max=self.num_embeddings - 1)
        return self.main(x)


class BinaryEncoder(nn.Module):
    def __init__(self, num_embeddings: int):
        super(BinaryEncoder, self).__init__()
        self.bit_num = num_embeddings
        self.main = nn.Embedding.from_pretrained(self.get_binary_embed_matrix(self.bit_num), freeze=True,
                                                 padding_idx=None)

    @staticmethod
    def get_binary_embed_matrix(bit_num):
        embedding_matrix = []
        for n in range(2 ** bit_num):
            embedding = [n >> d & 1 for d in range(bit_num)][::-1]
            embedding_matrix.append(embedding)
        return torch.tensor(embedding_matrix, dtype=torch.float)

    def forward(self, x: torch.Tensor):
        x = x.long().clamp_(max=2 ** self.bit_num - 1)
        return self.main(x)


class SignBinaryEncoder(nn.Module):
    def __init__(self, num_embeddings):
        super(SignBinaryEncoder, self).__init__()
        self.bit_num = num_embeddings
        self.main = nn.Embedding.from_pretrained(self.get_sign_binary_matrix(self.bit_num), freeze=True,
                                                 padding_idx=None)
        self.max_val = 2 ** (self.bit_num - 1) - 1

    @staticmethod
    def get_sign_binary_matrix(bit_num):
        neg_embedding_matrix = []
        pos_embedding_matrix = []
        for n in range(1, 2 ** (bit_num - 1)):
            embedding = [n >> d & 1 for d in range(bit_num - 1)][::-1]
            neg_embedding_matrix.append([1] + embedding)
            pos_embedding_matrix.append([0] + embedding)
        embedding_matrix = neg_embedding_matrix[::-1] + [[0 for _ in range(bit_num)]] + pos_embedding_matrix
        return torch.tensor(embedding_matrix, dtype=torch.float)

    def forward(self, x: torch.Tensor):
        x = x.long().clamp_(max=self.max_val, min=- self.max_val)
        return self.main(x + self.max_val)


class PositionEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=None):
        super(PositionEncoder, self).__init__()
        self.n_position = num_embeddings
        self.embedding_dim = self.n_position if embedding_dim is None else embedding_dim
        self.position_enc = nn.Embedding.from_pretrained(
            self.position_encoding_init(self.n_position, self.embedding_dim),
            freeze=True, padding_idx=None)

    @staticmethod
    def position_encoding_init(n_position, embedding_dim):
        ''' Init the sinusoid position encoding table '''

        # keep dim 0 for padding token position encoding zero vector
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / embedding_dim) for j in range(embedding_dim)]
            for pos in range(n_position)])
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # apply sin on 0th,2nd,4th...embedding_dim
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # apply cos on 1st,3rd,5th...embedding_dim
        return torch.from_numpy(position_enc).type(torch.FloatTensor)

    def forward(self, x: torch.Tensor):
        return self.position_enc(x)


class TimeEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(TimeEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.position_array = torch.nn.Parameter(self.get_position_array(), requires_grad=False)

    def get_position_array(self):
        x = torch.arange(0, self.embedding_dim, dtype=torch.float)
        x = x // 2 * 2
        x = torch.div(x, self.embedding_dim)
        x = torch.pow(10000., x)
        x = torch.div(1., x)
        return x

    def forward(self, x: torch.Tensor):
        v = torch.zeros(size=(x.shape[0], self.embedding_dim), dtype=torch.float, device=x.device)
        assert len(x.shape) == 1
        x = x.unsqueeze(dim=1)
        v[:, 0::2] = torch.sin(x * self.position_array[0::2])  # even
        v[:, 1::2] = torch.cos(x * self.position_array[1::2])  # odd
        return v


class UnsqueezeEncoder(nn.Module):
    def __init__(self, unsqueeze_dim: int = -1, norm_value: float = 1):
        super(UnsqueezeEncoder, self).__init__()
        self.unsqueeze_dim = unsqueeze_dim
        self.norm_value = norm_value

    def forward(self, x: torch.Tensor):
        x = x.float().unsqueeze(dim=self.unsqueeze_dim)
        if self.norm_value != 1:
            x = x / self.norm_value
        return x

class ScalarEncoder(nn.Module):
    def __init__(self, cfg):
        super(ScalarEncoder, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model.scalar_encoder
        self.output_size = self.cfg.output_dim
        self.encode_modules = nn.ModuleDict()
        for k, item in self.cfg.modules.items():
            if item['arc'] == 'time':
                self.encode_modules[k] = TimeEncoder(embedding_dim=item['embedding_dim'])
            elif item['arc'] == 'one_hot':
                self.encode_modules[k] = OnehotEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'binary':
                self.encode_modules[k] = BinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'sign_binary':
                self.encode_modules[k] = SignBinaryEncoder(num_embeddings=item['num_embeddings'], )
            else:
                print(f'cant implement {k} for arc {item["arc"]}')
                raise NotImplementedError

        self.layers = MLP(in_channels=self.cfg.input_dim, hidden_channels=self.cfg.hidden_dim,
                          out_channels=self.cfg.output_dim,
                          layer_num=self.cfg.layer_num,
                          layer_fn=fc_block,
                          activation=self.cfg.activation,
                          norm_type=self.cfg.norm_type,
                          use_dropout=False
                          )

    def forward(self, x: Dict[str, Tensor]):
        embeddings = []
        for key, item in self.cfg.modules.items():
            assert key in x, key
            embeddings.append(self.encode_modules[key](x[key]))

        out = torch.cat(embeddings, dim=-1)
        out = self.layers(out)
        return out


class TeamEncoder(nn.Module):
    def __init__(self, cfg):
        super(TeamEncoder, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model.team_encoder
        self.encode_modules = nn.ModuleDict()

        for k, item in self.cfg.modules.items():
            if item['arc'] == 'one_hot':
                self.encode_modules[k] = OnehotEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'binary':
                self.encode_modules[k] = BinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'sign_binary':
                self.encode_modules[k] = SignBinaryEncoder(num_embeddings=item['num_embeddings'], )
            else:
                print(f'cant implement {k} for arc {item["arc"]}')
                raise NotImplementedError

        self.embedding_dim = self.cfg.embedding_dim
        self.encoder_cfg = self.cfg.encoder
        self.encode_layers = MLP(in_channels=self.encoder_cfg.input_dim,
                                 hidden_channels=self.encoder_cfg.hidden_dim,
                                 out_channels=self.embedding_dim,
                                 layer_num=self.encoder_cfg.layer_num,
                                 layer_fn=fc_block,
                                 activation=self.encoder_cfg.activation,
                                 norm_type=self.encoder_cfg.norm_type,
                                 use_dropout=False)
        # self.activation_type = self.cfg.activation

        self.transformer_cfg = self.cfg.transformer
        self.transformer = Transformer(
            n_heads=self.transformer_cfg.head_num,
            embedding_size=self.embedding_dim,
            ffn_size=self.transformer_cfg.ffn_size,
            n_layers=self.transformer_cfg.layer_num,
            attention_dropout=0.0,
            relu_dropout=0.0,
            dropout=0.0,
            activation=self.transformer_cfg.activation,
            variant=self.transformer_cfg.variant,
        )
        self.output_cfg = self.cfg.output
        self.output_size = self.output_cfg.output_dim
        self.output_fc = fc_block(self.embedding_dim,
                                  self.output_cfg.output_dim,
                                  norm_type=self.output_cfg.norm_type,
                                  activation=self.output_cfg.activation)

    def forward(self, x):
        embeddings = []
        player_num = x['player_num']
        mask = sequence_mask(player_num, max_len=x['view_x'].shape[1])
        for key, item in self.cfg.modules.items():
            assert key in x, f"{key} not implemented"
            x_input = x[key]
            embeddings.append(self.encode_modules[key](x_input))

        x = torch.cat(embeddings, dim=-1)
        x = self.encode_layers(x)
        x = self.transformer(x, mask=mask)
        team_info = self.output_fc(x.sum(dim=1) / player_num.unsqueeze(dim=-1))
        return team_info


class BallEncoder(nn.Module):
    def __init__(self, cfg):
        super(BallEncoder, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model.ball_encoder
        self.encode_modules = nn.ModuleDict()
        for k, item in self.cfg.modules.items():
            if item['arc'] == 'one_hot':
                self.encode_modules[k] = OnehotEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'binary':
                self.encode_modules[k] = BinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'sign_binary':
                self.encode_modules[k] = SignBinaryEncoder(num_embeddings=item['num_embeddings'], )
            elif item['arc'] == 'unsqueeze':
                self.encode_modules[k] = UnsqueezeEncoder()
            else:
                print(f'cant implement {k} for arc {item["arc"]}')
                raise NotImplementedError
        self.embedding_dim = self.cfg.embedding_dim
        self.encoder_cfg = self.cfg.encoder
        self.encode_layers = MLP(in_channels=self.encoder_cfg.input_dim,
                                 hidden_channels=self.encoder_cfg.hidden_dim,
                                 out_channels=self.embedding_dim,
                                 layer_num=self.encoder_cfg.layer_num,
                                 layer_fn=fc_block,
                                 activation=self.encoder_cfg.activation,
                                 norm_type=self.encoder_cfg.norm_type,
                                 use_dropout=False)

        self.transformer_cfg = self.cfg.transformer
        self.transformer = Transformer(
            n_heads=self.transformer_cfg.head_num,
            embedding_size=self.embedding_dim,
            ffn_size=self.transformer_cfg.ffn_size,
            n_layers=self.transformer_cfg.layer_num,
            attention_dropout=0.0,
            relu_dropout=0.0,
            dropout=0.0,
            activation=self.transformer_cfg.activation,
            variant=self.transformer_cfg.variant,
        )
        self.output_cfg = self.cfg.output
        self.output_size = self.output_cfg.output_dim
        self.output_fc = fc_block(self.embedding_dim,
                                  self.output_cfg.output_dim,
                                  norm_type=self.output_cfg.norm_type,
                                  activation=self.output_cfg.activation)

    def forward(self, x):
        ball_num = x['ball_num']
        embeddings = []
        mask = sequence_mask(ball_num, max_len=x['x'].shape[1])
        for key, item in self.cfg.modules.items():
            assert key in x, key
            x_input = x[key]
            embeddings.append(self.encode_modules[key](x_input))
        x = torch.cat(embeddings, dim=-1)
        x = self.encode_layers(x)
        x = self.transformer(x, mask=mask)

        ball_info = x.sum(dim=1) / ball_num.unsqueeze(dim=-1)
        ball_info = self.output_fc(ball_info)
        return x, ball_info


class SpatialEncoder(nn.Module):
    def __init__(self, cfg):
        super(SpatialEncoder, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model.spatial_encoder

        # scatter related
        self.spatial_x = 64
        self.spatial_y = 64
        self.scatter_cfg = self.cfg.scatter
        self.scatter_fc = fc_block(in_channels=self.scatter_cfg.input_dim, out_channels=self.scatter_cfg.output_dim,
                                   activation=self.scatter_cfg.activation, norm_type=self.scatter_cfg.norm_type)
        self.scatter_connection = ScatterConnection(self.scatter_cfg.scatter_type)

        # resnet related
        self.resnet_cfg = self.cfg.resnet
        self.get_resnet_blocks()

        self.output_cfg = self.cfg.output
        self.output_size = self.output_cfg.output_dim
        self.output_fc = fc_block(
            in_channels=self.spatial_x // 8 * self.spatial_y // 8 * self.resnet_cfg.down_channels[-1],
            out_channels=self.output_cfg.output_dim,
            norm_type=self.output_cfg.norm_type,
            activation=self.output_cfg.activation)

    def get_resnet_blocks(self):
        # 2 means food/spore embedding
        project = conv2d_block(in_channels=self.scatter_cfg.output_dim + 2,
                               out_channels=self.resnet_cfg.project_dim,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               activation=self.resnet_cfg.activation,
                               norm_type=self.resnet_cfg.norm_type,
                               bias=False,
                               )

        layers = [project]
        dims = [self.resnet_cfg.project_dim] + self.resnet_cfg.down_channels
        for i in range(len(dims) - 1):
            layer = conv2d_block(in_channels=dims[i],
                                 out_channels=dims[i + 1],
                                 kernel_size=4,
                                 stride=2,
                                 padding=1,
                                 activation=self.resnet_cfg.activation,
                                 norm_type=self.resnet_cfg.norm_type,
                                 bias=False,
                                 )
            layers.append(layer)
            layers.append(ResBlock(in_channels=dims[i + 1],
                                   activation=self.resnet_cfg.activation,
                                   norm_type=self.resnet_cfg.norm_type))
        self.resnet = torch.nn.Sequential(*layers)


    def get_background_embedding(self, coord_x, coord_y, num, ):

        background_ones = torch.ones(size=(coord_x.shape[0], coord_x.shape[1]), device=coord_x.device)
        background_mask = sequence_mask(num, max_len=coord_x.shape[1])
        background_ones = (background_ones * background_mask).unsqueeze(-1)
        background_embedding = self.scatter_connection.xy_forward(background_ones,
                                                                  spatial_size=[self.spatial_x, self.spatial_y],
                                                                  coord_x=coord_x,
                                                                  coord_y=coord_y)

        return background_embedding

    def forward(self, inputs, ball_embeddings, ):
        spatial_info = inputs['spatial_info']
        # food and spore
        food_embedding = self.get_background_embedding(coord_x=spatial_info['food_x'],
                                                       coord_y=spatial_info['food_y'],
                                                       num=spatial_info['food_num'], )

        spore_embedding = self.get_background_embedding(coord_x=spatial_info['spore_x'],
                                                        coord_y=spatial_info['spore_y'],
                                                        num=spatial_info['spore_num'], )
        # scatter ball embeddings
        ball_info = inputs['ball_info']
        ball_num = ball_info['ball_num']
        ball_mask = sequence_mask(ball_num, max_len=ball_embeddings.shape[1])
        ball_embedding = self.scatter_fc(ball_embeddings) * ball_mask.unsqueeze(dim=2)

        ball_embedding = self.scatter_connection.xy_forward(ball_embedding,
                                                            spatial_size=[self.spatial_x, self.spatial_y],
                                                            coord_x=spatial_info['ball_x'],
                                                            coord_y=spatial_info['ball_y'])

        x = torch.cat([food_embedding, spore_embedding, ball_embedding], dim=1)

        x = self.resnet(x)

        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.output_fc(x)
        return x


class Encoder(nn.Module):
    def __init__(self, cfg, args):
        super(Encoder, self).__init__()
        self.whole_cfg = cfg
        self.scalar_encoder = ScalarEncoder(cfg)
        self.team_encoder = TeamEncoder(cfg)
        self.ball_encoder = BallEncoder(cfg)
        self.spatial_encoder = SpatialEncoder(cfg)
        embedding_dim = (self.scalar_encoder.output_size + self.team_encoder.output_size \
            + self.ball_encoder.output_size + self.spatial_encoder.output_size) * args.player_num_per_team
        self.output_fc = fc_block(embedding_dim,
                                  args.hidden_size,
                                  norm_type='LN',
                                  activation='relu')
        self.output_size = args.hidden_size
        

    def forward(self, x):
        scalar_info = self.scalar_encoder(x['scalar_info'])
        team_info = self.team_encoder(x['team_info'])
        ball_embeddings, ball_info = self.ball_encoder(x['ball_info'])
        spatial_info = self.spatial_encoder(x, ball_embeddings)
        x = self.output_fc(torch.flatten(torch.cat([scalar_info, team_info, ball_info, spatial_info], dim=1)))
        return x
