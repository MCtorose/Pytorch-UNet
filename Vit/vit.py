import torch
from vit_pytorch import ViT

# 定义ViT模型
v = ViT(
    image_size=(224, 256),
    patch_size=16,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
)

img = torch.randn(1, 3, 224, 256)
preds = v(img)
print(preds)
print(preds.shape)

# from collections import OrderedDict
# from functools import partial
#
# import torch
# import torch.nn as nn
# from vit_pytorch import ViT
# from vit_pytorch.cct import DropPath
#
#
# class PatchEmbed(nn.Module):
#     """
#     Image --> Patch Embedding --> Linear Proj --> Pos Embedding
#     Image size -> [224,224,3]
#     Patch size -> 16*16
#     Patch num -> (224^2)/(16^2)=196
#     Patch dim -> 16*16*3 =768
#     Patch Embedding: [224,224,3] -> [196,768]
#     Linear Proj: [196,768] -> [196,768]
#  	Positional Embedding: [197,768] -> [196,768]
#     """
#
#     def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
#         """
#         Args:
#             img_size: 默认参数224
#             patch_size: 默认参数是16
#             in_c: 输入的通道数,默认为3
#             embed_dim: 16*16*3 = 768
#             norm_layer: 是否使用norm层，默认为否
#         """
#         super().__init__()
#         img_size = (img_size, img_size)  # -> img_size = (224,224)
#         patch_size = (patch_size, patch_size)  # -> patch_size = (16,16)
#         self.img_size = img_size  # -> (224,224)
#         self.patch_size = patch_size  # -> (16,16)
#         self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # -> grid_size = (14,14)
#         self.num_patches = self.grid_size[0] * self.grid_size[1]  # -> num_patches = 196
#         # Patch+linear proj的这个操作 [224,224,3] --> [14,14,768]
#         self.proj = nn.Conv2d(in_channels=in_c, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
#         # 判断是否有norm_layer层，要是没有不改变输入
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#
#     def forward(self, x):
#         # 计算各个维度的大小
#         B, C, H, W = x.shape
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#
#         # flatten: [B, C, H, W] -> [B, C, HW], flatten(2)代表的是从2位置开始展开
#         # eg: [1,3,224,224] --> [1,768,14,14] -flatten->[1,768,196]
#         # transpose: [B, C, HW] -> [B, HW, C]
#         # eg: [1,768,196] -transpose-> [1,196,768]
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         x = self.norm(x)
#         return x
#
#
# class Attention(nn.Module):
#     def __init__(self,
#                  dim,  # 输入token的dim
#                  num_heads=8,  # attention head的个数
#                  qkv_bias=False,  # 是否使用qkv bias
#                  qk_scale=None,
#                  attn_drop_ratio=0.,
#                  proj_drop_ratio=0.):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         # 计算每一个head处理的维度head_dim = dim // num_heads --> 768/8 = 96
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5  # 根下dk操作
#         # 使用nn.Linear生成w_q,w_k,w_v，因为本质上每一个变换矩阵都是线性变换
#         self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop_ratio)
#         self.proj = nn.Linear(in_features=dim, out_features=dim)
#         self.proj_drop = nn.Dropout(proj_drop_ratio)
#
#     def forward(self, x):
#         """
#         Attention(
#               (qkv): Linear(in_features=768, out_features=2304, bias=True)
#               (attn_drop): Dropout(p=0.0, inplace=False)
#               (proj): Linear(in_features=768, out_features=768, bias=True)
#               (proj_drop): Dropout(p=0.0, inplace=False)
#         )
#         """
#
#         # [batch_size, num_patches + 1, total_embed_dim]
#         # total_embed_dim不是一开始展开的那个维度，是经过了一个线性变换层得到的
#         B, N, C = x.shape
#
#         # [batch_size, num_patches+1, total_embed_dim] -qkv()-> [batch_size, num_patches + 1, 3 * total_embed_dim]
#         # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]  排布规律：内存连续性，优先行排列，类似于多维数组循环排布
#         # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         # q,k,v = [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#
#         # transpose(-2,-1)在最后两个维度进行操作，输入的形状[batch_size,num_heads,num_patches+1,embed_dim_per_head]
#         # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
#         # @: multiply 矩阵叉乘 -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#         # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
#         # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
#
# class Mlp(nn.Module):
#     """多层感知机（Multilayer Perceptron, MLP）是一种前馈神经网络，它包含一个或多个隐藏层，每个隐藏层由多个神经元（或称为节点）组成。
#     in_features --> hidden_features --> out_features
#     论文实现时：in_features.shape = out_features.shape
#     """  # MLP 的主要作用是学习输入数据和输出数据之间的复杂映射关系，通过多层非线性变换来捕捉数据中的高级特征和模式。MLP 可以用于分类、回归、特征提取等多种机器学习任务。
#
#     # 隐藏层的 hidden_features 数量则需要根据模型的复杂度和训练数据的大小来平衡，以避免过拟合或欠拟合。
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         # 用or实现了或操作，当hidden_features/out_features为默认值None时
#         # 此时out_features/hidden_features=None or in_features = in_features
#         # 当对out_features或hidden_features进行输入时，or操作将会默认选择or前面的
#         # 此时out_features/hidden_features = out_features/hidden_features
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         # in_features --> hidden_features --> out_features
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
#
#
# class Block(nn.Module):
#     """
#     每一个Encoder Block的构成
#     每个Encode Block的流程：norm1 --> Multi-Head Attention --> norm2 --> MLP
#     """
#
#     def __init__(self,
#                  dim,  # 输入mlp的维度
#                  num_heads,  # Multi-Head-Attention的头个数
#                  mlp_ratio=4.,  # hidden_features / in_features = mlp_ratio
#                  qkv_bias=False,  # q,k,v的生成是否使用bias
#                  qk_scale=None,
#                  drop_ratio=0.,  # dropout的比例
#                  attn_drop_ratio=0.,  # 注意力dropout的比例
#                  drop_path_ratio=0.,
#                  act_layer=nn.GELU,  # 激活函数默认使用GELU
#                  norm_layer=nn.LayerNorm):  # Norm默认使用LayerNorm
#         super(Block, self).__init__()
#         # 第一层normalization
#         self.norm1 = norm_layer(dim)
#         # self.attention层的实现
#         self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
#         self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
#         # 第二层normalization
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)  # hidden_dim = dim * mlp_ratio
#         # mlp实现
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
#
#     def forward(self, x):
#         # 实现了两个残差连接
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x
#
#
# class VisionTransformer(nn.Module):
#     def __init__(self,
#                  img_size=224,
#                  patch_size=16,
#                  in_c=3,
#                  num_classes=1000,
#                  embed_dim=768,
#                  depth=12,
#                  num_heads=12,
#                  mlp_ratio=4.0,
#                  qkv_bias=True,
#                  qk_scale=None,
#                  representation_size=None,
#                  distilled=False,
#                  drop_ratio=0.,
#                  attn_drop_ratio=0.,
#                  drop_path_ratio=0.,
#                  embed_layer=PatchEmbed,
#                  norm_layer=None,
#                  act_layer=None):
#         """
#         Args:
#             img_size (int, tuple): input image size
#             patch_size (int, tuple): patch size
#             in_c (int): number of input channels
#             num_classes (int): number of classes for classification head
#             embed_dim (int): embedding dimension
#             depth (int): depth of transformer
#             num_heads (int): number of attention heads
#             mlp_ratio (int): ratio of mlp hidden dim to embedding dim
#             qkv_bias (bool): enable bias for qkv if True
#             qk_scale (float): override default qk scale of head_dim ** -0.5 if set
#             representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
#             distilled (bool): model includes a distillation token and head as in DeiT models
#             drop_ratio (float): dropout rate
#             attn_drop_ratio (float): attention dropout rate
#             drop_path_ratio (float): stochastic depth rate 随机深度概率
#             embed_layer (nn.Module): patch embedding layer
#             norm_layer: (nn.Module): normalization layer
#         """
#         super(VisionTransformer, self).__init__()
#         self.num_classes = num_classes
#         # 每个patch的图像维度 = embed_dim
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#         # token的个数为1
#         self.num_tokens = 2 if distilled else 1
#         # 设置激活函数和norm函数
#         norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
#         act_layer = act_layer or nn.GELU
#         # 对应的将图片打成patch的操作
#         self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
#         num_patches = self.patch_embed.num_patches
#         # 设置分类的cls_token
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # torch.Size([1, 1, 768])
#         # distilled 是Deit中的 这里为None
#         self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
#         # pos_embedding 为一个可以学习的参数
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))  # torch.Size([1, 197, 768])
#         self.pos_drop = nn.Dropout(p=drop_ratio)
#
#         dpr = [x.item() for x in torch.linspace(start=0, end=drop_path_ratio, steps=depth)]  # stochastic depth decay rule
#         # 使用nn.Sequential进行构建，ViT中深度为12
#         self.blocks = nn.Sequential(*[
#             Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                   drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
#                   norm_layer=norm_layer, act_layer=act_layer)
#             for i in range(depth)
#         ])
#         self.norm = norm_layer(embed_dim)
#
#         # Representation layer
#         if representation_size and not distilled:
#             self.has_logits = True
#             self.num_features = representation_size
#             self.pre_logits = nn.Sequential(OrderedDict([
#                 ("fc", nn.Linear(embed_dim, representation_size)),
#                 ("act", nn.Tanh())
#             ]))
#         else:
#             self.has_logits = False
#             self.pre_logits = nn.Identity()
#
#         # Classifier head(s) 分类层
#         self.head = nn.Linear(in_features=self.num_features, out_features=num_classes) if num_classes > 0 else nn.Identity()
#         self.head_dist = None
#         if distilled:
#             self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
#
#         # Weight init 初始化位置编码  nn.init.trunc_normal_ 是一个用于初始化张量的函数，它生成截断正态分布（truncated normal distribution）的随机数来填充指定的张量。
#         nn.init.trunc_normal_(tensor=self.pos_embed, std=0.02)
#         if self.dist_token is not None:
#             nn.init.trunc_normal_(self.dist_token, std=0.02)
#
#         nn.init.trunc_normal_(self.cls_token, std=0.02)
#         self.apply(_init_vit_weights)  # self.apply 是一个方法，它递归地对模型的所有子模块应用一个函数。
#
#     def forward_features(self, x):
#         # [B, C, H, W] -> [B, num_patches, embed_dim]
#         x = self.patch_embed(x)  # [B, 196, 768]
#         # [1, 1, 768] -> [B, 1, 768]
#         cls_token = self.cls_token.expand(x.shape[0], -1, -1)
#         if self.dist_token is None:
#             x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
#         else:
#             x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
#
#         x = self.pos_drop(x + self.pos_embed)
#         x = self.blocks(x)
#         x = self.norm(x)
#         if self.dist_token is None:
#             return self.pre_logits(x[:, 0])
#         else:
#             return x[:, 0], x[:, 1]
#
#     def forward(self, x):
#         x = self.forward_features(x)
#         if self.head_dist is not None:
#             x, x_dist = self.head(x[0]), self.head_dist(x[1])
#             if self.training and not torch.jit.is_scripting():
#                 # during inference, return the average of both classifier predictions
#                 return x, x_dist
#             else:
#                 return (x + x_dist) / 2
#         else:
#             x = self.head(x)
#         return x
#
#
# def _init_vit_weights(m):
#     """
#     ViT weight initialization
#     :param m: module
#     """
#     if isinstance(m, nn.Linear):
#         nn.init.trunc_normal_(m.weight, std=.01)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)
#     elif isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, mode="fan_out")
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)
#     elif isinstance(m, nn.LayerNorm):
#         nn.init.zeros_(m.bias)
#         nn.init.ones_(m.weight)
#
#
# if __name__ == '__main__':
#     model = VisionTransformer(img_size=224, patch_size=16, embed_dim=16 * 16 * 3, depth=12, num_heads=12, mlp_ratio=4)
#     print(model)
#     input = torch.randn(1, 3, 224, 224)
#     res = model(input)
#     print(res)
#     print(res.shape)
