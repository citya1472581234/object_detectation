# object_detectation
* yolo layer 
--------------

    grid_x = torch.linspace(0, g_dim-1, g_dim).repeat(g_dim,1).repeat(bs*self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
    grid_y = torch.linspace(0, g_dim-1, g_dim).repeat(g_dim,1).t().repeat(bs*self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)
    scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]
    anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
    anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
    anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, g_dim*g_dim).view(w.shape)
    anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, g_dim*g_dim).view(h.shape)
    製作網格的位置，預測的x,y,w,h都是小於1的，所以我們在加上網格位置，例如 : dim=9，網格 9x9 ，位置為0-9。
   

* build target
 -------------
 
    mask        = torch.zeros(nB, nA, dim, dim)
    conf_mask   = torch.ones(nB, nA, dim, dim)
    tx          = torch.zeros(nB, nA, dim, dim)
    ty          = torch.zeros(nB, nA, dim, dim)
    tw          = torch.zeros(nB, nA, dim, dim)
    th          = torch.zeros(nB, nA, dim, dim)
    tconf       = torch.zeros(nB, nA, dim, dim)
    tcls        = torch.zeros(nB, nA, dim, dim, num_classes)
    
    
    gx = target[b, t, 1] * dim
    gy = target[b, t, 2] * dim
    gw = target[b, t, 3] * dim
    gh = target[b, t, 4] * dim
    gi = int(gx)
    gj = int(gy)
    gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
    anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1)
    anch_ious = bbox_iou(gt_box, anchor_shapes)
    conf_mask[b, anch_ious > ignore_thres] = 0
    conf_mask[nB, nA] = 0 把跟anchor boxes 大於閥值的設為0 
    best_n = np.argmax(anch_ious)
    gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
    pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
    mask[b, best_n, gj, gi] = 1
    conf_mask[b, best_n, gj, gi] = 1
    再把best_n位置的網格設為 1 ，類似NMS非極大抑制效果刪除過多預測，留下IOU最大的一個
    tx[b, best_n, gj, gi] = gx - gi
    ty[b, best_n, gj, gi] = gy - gj
    我們將 gx - gi、gy - gj ，所以tx ty為0-1的值
    tw[b, best_n, gj, gi] = math.log(gw/anchors[best_n][0] + 1e-16)
    th[b, best_n, gj, gi] = math.log(gh/anchors[best_n][1] + 1e-16)
    tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1
    iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
    tconf[b, best_n, gj, gi] = 1

* loss function
---------------
     
     loss_x = self.lambda_coord * self.bce_loss(x * mask, tx * mask)
     loss_y = self.lambda_coord * self.bce_loss(y * mask, ty * mask)
     loss_w = self.lambda_coord * self.mse_loss(w * mask, tw * mask) / 2
     loss_h = self.lambda_coord * self.mse_loss(h * mask, th * mask) / 2
     loss_conf = self.bce_loss(conf * conf_mask, tconf * conf_mask)
     loss_cls = self.bce_loss(pred_cls * cls_mask, tcls * cls_mask)
     
     
