Timer unit: 1e-06 s

Total time: 28503.7 s
File: /home/ubuntu/anaconda3/envs/dgl/lib/python3.7/site-packages/torch/autograd/grad_mode.py
Function: decorate_context at line 23

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    23                                                   @functools.wraps(func)
    24                                                   def decorate_context(*args, **kwargs):
    25      1233      55046.0     44.6      0.0              with self.__class__():
    26      1233 28503647494.0 23117313.5    100.0                  return func(*args, **kwargs)

Total time: 1.62745 s
File: /home/ubuntu/workspace/KPConv-dgl/dataset.py
Function: get_bbox at line 9

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     9                                           @profile
    10                                           def get_bbox(points):
    11                                               r"""
    12                                               
    13                                               Description
    14                                               -----------
    15                                               Get the bounding box of a point cloud
    16                                               
    17                                               Parameters
    18                                               ----------
    19                                               points: torch.Tensor
    20                                                   [N, 3] matrix of input points
    21                                               """
    22     15082    1057197.0     70.1     65.0      min_point = torch.min(points, 0)[0].data
    23     15082     561588.0     37.2     34.5      max_point = torch.max(points, 0)[0].data
    24                                               
    25     15082       8668.0      0.6      0.5      return min_point, max_point

Total time: 40377.2 s
File: /home/ubuntu/workspace/KPConv-dgl/dataset.py
Function: grid_subsampling at line 28

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    28                                           @profile
    29                                           def grid_subsampling(points, feats, dl):
    30                                               r"""
    31                                           
    32                                               Description
    33                                               -----------
    34                                               Grid subsampling is implemented by C++ in the author's PyTorch code
    35                                               
    36                                               Parameters
    37                                               ----------
    38                                               points: torch.Tensor
    39                                                   [N, 3] matrix of input points
    40                                               feats: torch.Tensor
    41                                                   [N, D] matrix of input features
    42                                               dl: float
    43                                                   the size of grid voxels
    44                                           
    45                                               Return
    46                                               ----------
    47                                               subsampled points and features
    48                                               """
    49     15082     251888.0     16.7      0.0      n_points = len(points)
    50     15082    1752713.0    116.2      0.0      min_corner, max_corner = get_bbox(points)
    51                                               # is this nessesary?
    52     15082    1181100.0     78.3      0.0      min_corner = torch.floor(torch.div(min_corner, dl)) * dl
    53                                               
    54     15082     678215.0     45.0      0.0      sample_nx = int((max_corner[0] - min_corner[0]) / dl) + 1
    55     15082     388683.0     25.8      0.0      sample_ny = int((max_corner[1] - min_corner[1]) / dl) + 1
    56                                           
    57     15082      43902.0      2.9      0.0      data = defaultdict(list)
    58                                           
    59 275296901 2499978175.0      9.1      6.2      for p, f in zip(points, feats):
    60 275281819 10734572147.0     39.0     26.6          idx_x = int((p[0] - min_corner[0]) / dl)
    61 275281819 9254710695.0     33.6     22.9          idx_y = int((p[1] - min_corner[1]) / dl)
    62 275281819 9135028318.0     33.2     22.6          idx_z = int((p[2] - min_corner[2]) / dl)
    63 275281819  334856970.0      1.2      0.8          idx = idx_x + sample_nx * idx_y + sample_nx * sample_ny * idx_z
    64 275281819 5608718755.0     20.4     13.9          data[idx].append(torch.cat([p, f]).view(1, -1))
    65                                               
    66     15082      42471.0      2.8      0.0      subsampled_data = []
    67                                               
    68  87758818   88597875.0      1.0      0.2      for _, v in data.items():
    69  87743736  899053619.0     10.2      2.2          v = torch.cat(v)
    70  87743736 1644015410.0     18.7      4.1          subsampled_data.append(torch.mean(v, dim=0).view(1, -1))
    71                                           
    72     15082  172465620.0  11435.2      0.4      subsampled_data = torch.cat(subsampled_data)
    73                                           
    74     15082     832041.0     55.2      0.0      return subsampled_data[:, :3], subsampled_data[:, 3:]

Total time: 4.95672 s
File: /home/ubuntu/workspace/KPConv-dgl/dataset.py
Function: collate_fn at line 77

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    77                                           @profile
    78                                           def collate_fn(batch):
    79                                               r"""
    80                                                   points: list, [b, n, 3]
    81                                                   feats: list, [b, n, d]
    82                                                   labels: list, [b,]
    83                                                   length: list, [b,], each value is the number of points in each point cloud
    84                                               """
    85      2771      28172.0     10.2      0.6      points, feats, labels, length = map(list, zip(*batch))
    86      2771    2585321.0    933.0     52.2      batch_points = torch.FloatTensor(np.concatenate(points))
    87      2771    2280354.0    822.9     46.0      batch_feats = torch.FloatTensor(np.concatenate(feats))
    88      2771      48928.0     17.7      1.0      batch_labels = torch.LongTensor(labels).view(-1, 1)
    89      2771      12052.0      4.3      0.2      batch_len = torch.LongTensor(length)
    90      2771       1891.0      0.7      0.0      return batch_points, batch_feats, batch_labels, batch_len

Total time: 15225.9 s
File: /home/ubuntu/workspace/KPConv-dgl/dataset.py
Function: __getitem__ at line 118

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   118                                               @profile
   119                                               def __getitem__(self, idx):
   120     22154      89431.0      4.0      0.0          if idx in self.cache:
   121      9843      17558.0      1.8      0.0              return self.cache[idx]
   122                                                   else:
   123     12311     146792.0     11.9      0.0              cloud_name = self.names[idx]
   124     12311      62709.0      5.1      0.0              class_folder = '_'.join(cloud_name.split('_')[:-1])
   125     12311      75265.0      6.1      0.0              txt_file = f'{self.root}/{class_folder}/{cloud_name}.txt'
   126                                                       # point cloud
   127     12311 2205520404.0 179150.4     14.5              data = np.loadtxt(txt_file, delimiter=',', dtype=np.float32)
   128     12311     558141.0     45.3      0.0              data = torch.FloatTensor(data)
   129     12311     224101.0     18.2      0.0              points, feats = grid_subsampling(points=data[:, :3],
   130     12311      55395.0      4.5      0.0                                               feats=data[:, 3:],
   131     12311 12993415672.0 1055431.4     85.3                                               dl=self.dl)
   132                                           
   133     12311      46028.0      3.7      0.0              if self.orient_correction:
   134     12311   12590002.0   1022.7      0.1                  points = points[:, [0, 2, 1]]
   135     12311   12552600.0   1019.6      0.1                  feats = feats[:, [0, 2, 1]]
   136                                                       # ground truth
   137     12311      45364.0      3.7      0.0              label = self.name_to_label[class_folder]
   138                                           
   139     12311      41639.0      3.4      0.0              if len(self.cache) < self.cache_size:
   140     12311     354201.0     28.8      0.0                  self.cache[idx] = (points, feats, label, len(points))
   141                                           
   142     12311      63730.0      5.2      0.0              return points, feats, label, len(points)

Total time: 7.4924 s
File: /home/ubuntu/workspace/KPConv-dgl/model.py
Function: forward at line 23

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    23                                               @profile
    24                                               def forward(self, g, feats):
    25      2771     241958.0     87.3      3.2          with g.local_scope():
    26      2771    6253684.0   2256.8     83.5              feats = self.kpconv(g, feats)
    27      2771     968362.0    349.5     12.9              feats = self.leaky_relu(self.bn(feats))
    28      2771      28400.0     10.2      0.4              return feats

Total time: 8.71598 s
File: /home/ubuntu/workspace/KPConv-dgl/model.py
Function: forward at line 71

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    71                                               @profile
    72                                               def forward(self, g, feats):
    73      2771     260120.0     93.9      3.0          with g.local_scope():
    74      2771    1485286.0    536.0     17.0              x = self.down_scaling(feats)
    75      2771    5214123.0   1881.7     59.8              x = self.kpconv(g, x)
    76      2771     528113.0    190.6      6.1              x = self.leaky_relu(self.bn(x))
    77      2771     608768.0    219.7      7.0              x = self.up_scaling(x)
    78      2771     619570.0    223.6      7.1              return self.leaky_relu(x + self.res(feats))

Total time: 47156.3 s
File: /home/ubuntu/workspace/KPConv-dgl/model.py
Function: forward at line 99

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    99                                               @profile
   100                                               def forward(self, points, feats, length):
   101      2771 17452980467.0 6298441.2     37.0          batch_g = self.fnn1(points, feats, length)
   102      2771    7745845.0   2795.3      0.0          feats = self.block1(batch_g, batch_g.ndata['feat'])
   103      2771 29583201369.0 10676001.9     62.7          points1, feats1, length1 = self.bgs(batch_g.ndata['pos'], feats, length)
   104                                                   
   105      2771  101624584.0  36674.3      0.2          batch_g1 = self.fnn2(points1, feats1, length1)
   106      2771    8859755.0   3197.3      0.0          feats1 = self.block3(batch_g1, feats1)
   107                                                   
   108      2771    1478255.0    533.5      0.0          g_feat = self.pool(batch_g1, feats1)
   109      2771     446758.0    161.2      0.0          return self.output(g_feat)

Total time: 17510.4 s
File: /home/ubuntu/workspace/KPConv-dgl/modules.py
Function: forward at line 30

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    30                                               @profile
    31                                               def forward(self, batch_points, batch_feats, batch_len):
    32      5542       6821.0      1.2      0.0          batch_g = []
    33      5542     922463.0    166.4      0.0          batch_len = torch.cat([torch.zeros(1).to(batch_len.device), torch.cumsum(batch_len, dim=0)])
    34                                           
    35     49850     163519.0      3.3      0.0          for i in range(len(batch_len) - 1):
    36     44308    2078432.0     46.9      0.0              current_cloud = batch_points[int(batch_len[i]):int(batch_len[i + 1])]
    37     44308     516347.0     11.7      0.0              num_points = len(current_cloud)
    38                                                       # get neighbors for each point
    39     44308   22553025.0    509.0      0.1              dists = square_distance(current_cloud)
    40     44308  730670890.0  16490.7      4.2              group_idx = torch.arange(num_points, dtype=torch.long).view(1, -1).repeat(num_points, 1)
    41     44308 13907266899.0 313877.1     79.4              group_idx[dists > self.radius ** 2] = num_points
    42                                                       # create undirected graph
    43     44308 2587310628.0  58393.8     14.8              src, dst = torch.where(group_idx != num_points)
    44     44308   28491713.0    643.0      0.2              g = dgl.graph((src, dst))
    45     44308  197490792.0   4457.2      1.1              g = dgl.to_bidirected(g)
    46     44308   10164293.0    229.4      0.1              g = g.to(batch_points.device)
    47     44308    3548388.0     80.1      0.0              g.ndata['pos'] = current_cloud
    48     44308    5698910.0    128.6      0.0              g.ndata['feat'] = batch_feats[int(batch_len[i]):int(batch_len[i + 1])]
    49     44308      69643.0      1.6      0.0              batch_g.append(g)
    50                                                   
    51      5542   13496860.0   2435.4      0.1          return dgl.batch(batch_g)

Total time: 29582.7 s
File: /home/ubuntu/workspace/KPConv-dgl/modules.py
Function: forward at line 63

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    63                                               @profile
    64                                               def forward(self, batch_points, batch_feats, batch_len):
    65                                                   r"""
    66                                                       assume that batch_points, batch_feats and batch_len are on the same device
    67                                                       +offset -> grid subsampling simultaneously -> -offset
    68                                                   """
    69      2771       4724.0      1.7      0.0          device = batch_points.device
    70      2771   78154321.0  28204.4      0.3          offsets = torch.arange(0, len(batch_len) * self.offset, self.offset).to(device)
    71                                                   # each offset will be repeated by the number of each point cloud
    72      2771     697636.0    251.8      0.0          batch_offsets = offsets.repeat_interleave(batch_len).reshape(-1, 1)  # [batch, 1]
    73      2771      95486.0     34.5      0.0          batch_offset_points = batch_points + batch_offsets  # [batch, 3]
    74                                                   
    75      2771 29397918791.0 10609137.1     99.4          pool_points, pool_feats = grid_subsampling(batch_offset_points, batch_feats, self.dl)
    76                                                   # calculate pool batch length
    77      2771    3303701.0   1192.2      0.0          tmp_points = torch.cat([pool_points, torch.zeros(1, pool_points.size()[1]).to(device)], dim=0)
    78                                                   # assume that there exists a gap between each point cloud
    79      2771    9548530.0   3445.9      0.0          gap = torch.abs(tmp_points[1:, :] - tmp_points[:-1, :]) >= self.offset - 2
    80      2771    1029238.0    371.4      0.0          pool_cumsum_batch = torch.cat([torch.zeros(1).to(device), torch.where(gap[:, 0] == True)[0] + 1])
    81      2771     155393.0     56.1      0.0          pool_batch_len = (pool_cumsum_batch[1:] - pool_cumsum_batch[:-1]).long()
    82                                                   # back to the origin scale
    83      2771   87153187.0  31451.9      0.3          pool_offsets = torch.arange(0, len(pool_batch_len) * self.offset, self.offset).to(device)
    84      2771    4319806.0   1558.9      0.0          pool_batch_offsets = pool_offsets.repeat_interleave(pool_batch_len).reshape(-1, 1)
    85      2771     355709.0    128.4      0.0          pool_points = pool_points - pool_batch_offsets
    86                                           
    87      2771       3897.0      1.4      0.0          return pool_points, pool_feats, pool_batch_len

Total time: 4.27901 s
File: /home/ubuntu/workspace/KPConv-dgl/modules.py
Function: msg_fn at line 116

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   116                                               @profile
   117                                               def msg_fn(self, edge):
   118      5542     848638.0    153.1     19.8          y = edge.src['pos'] - edge.dst['pos']  # centerize every neighborhood
   119      5542     334465.0     60.4      7.8          y = y.unsqueeze(1) - self.kernel_points  # [n_edges, K, p_dim]
   120      5542    1561340.0    281.7     36.5          h = self.relu(1 - torch.sqrt(torch.sum(y ** 2, dim=-1)) / self.KP_extent)  # [n_edges, K]
   121      5542      75903.0     13.7      1.8          h = h.unsqueeze(-1).unsqueeze(-1)  # [n_edges, K, 1, 1]
   122      5542     711910.0    128.5     16.6          m = torch.sum(h * self.weights, dim=1)  # [n_edges, K, in_dim, out_dim] -> [n_edges, in_dim, out_dim]
   123      5542     223052.0     40.2      5.2          f = edge.src['feat'].unsqueeze(1)  # [n_edges, 1, in_dim]
   124      5542     523700.0     94.5     12.2          return {'m': (f @ m).squeeze(1)}

Total time: 11.2437 s
File: /home/ubuntu/workspace/KPConv-dgl/modules.py
Function: forward at line 126

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   126                                               @profile
   127                                               def forward(self, g, feats):
   128      5542     253563.0     45.8      2.3          with g.local_scope():
   129      5542     368368.0     66.5      3.3              g.ndata['feat'] = feats
   130      5542   10442574.0   1884.3     92.9              g.update_all(self.msg_fn, fn.sum('m', 'h'))
   131      5542     179178.0     32.3      1.6              return g.ndata['h']

Total time: 35432.5 s
File: main.py
Function: train at line 11

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    11                                           @profile
    12                                           def train(model, device, data_loader, opt, loss_fn):
    13         1        270.0    270.0      0.0      model.train()
    14                                           
    15         1          1.0      1.0      0.0      train_loss = []
    16      1232 12325572107.0 10004522.8     34.8      for points, feats, labels, length in data_loader:
    17      1231     476941.0    387.4      0.0          points = points.to(device)
    18      1231     339967.0    276.2      0.0          feats = feats.to(device)
    19      1231      49336.0     40.1      0.0          labels = labels.to(device)
    20      1231      36588.0     29.7      0.0          length = length.to(device)
    21                                           
    22      1231 21569179272.0 17521672.8     60.9          logits = model(points, feats, length)
    23      1231  211293750.0 171644.0      0.6          loss = loss_fn(logits, labels.view(-1))
    24      1231     125610.0    102.0      0.0          train_loss.append(loss.item())
    25                                                   
    26      1231     932319.0    757.4      0.0          opt.zero_grad()
    27      1231 1315606066.0 1068729.5      3.7          loss.backward()
    28      1231    8866953.0   7203.0      0.0          opt.step()
    29                                               
    30         1        280.0    280.0      0.0      return sum(train_loss) / len(train_loss)

Total time: 63930.6 s
File: main.py
Function: main at line 53

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    53                                           @profile
    54                                           def main():
    55                                               # check cuda
    56         1      38162.0  38162.0      0.0      device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'
    57                                           
    58                                               # load dataset
    59         1      79331.0  79331.0      0.0      train_dataset = ModelNet40Dataset('data/ModelNet40', dl=args.dl, split='train')
    60         1      20356.0  20356.0      0.0      test_dataset = ModelNet40Dataset('data/ModelNet40', dl=args.dl, split='test')
    61                                               
    62         1        101.0    101.0      0.0      train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    63         1         50.0     50.0      0.0      test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    64                                           
    65                                               # load model
    66         1    2989392.0 2989392.0      0.0      model = KPCNN(args).to(device)
    67                                           
    68         1        676.0    676.0      0.0      logger.info(model)
    69                                           
    70         1        129.0    129.0      0.0      loss_fn = nn.CrossEntropyLoss()
    71         1        710.0    710.0      0.0      opt = optim.Adam(model.parameters(), lr=args.lr)
    72                                           
    73         1        128.0    128.0      0.0      logger.info('---------- Training ----------')
    74         2          6.0      3.0      0.0      for i in range(args.epochs):
    75         1 35432582869.0 35432582869.0     55.4          train_loss = train(model, device, train_loader, opt, loss_fn)
    76         1 20355298341.0 20355298341.0     31.8          train_acc = test(model, device, train_loader)
    77         1        310.0    310.0      0.0          logger.info(f'Epoch {i} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    78                                               
    79         1         85.0     85.0      0.0      logger.info('---------- Testing ----------')
    80         1 8139569014.0 8139569014.0     12.7      test_acc = test(model, device, test_loader)
    81         1        439.0    439.0      0.0      logger.info(f'Test Acc: {test_acc:.4f}')

