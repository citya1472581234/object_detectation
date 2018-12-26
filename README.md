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
    製作網格的位置，預測的x,y,w,h都是小於1的(需驗證)
    

* build target
 -------------
 
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
    best_n = np.argmax(anch_ious)
    gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
    pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
    mask[b, best_n, gj, gi] = 1
    conf_mask[b, best_n, gj, gi] = 1
    tx[b, best_n, gj, gi] = gx - gi
    ty[b, best_n, gj, gi] = gy - gj
    tw[b, best_n, gj, gi] = math.log(gw/anchors[best_n][0] + 1e-16)
    th[b, best_n, gj, gi] = math.log(gh/anchors[best_n][1] + 1e-16)
    tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1
    iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
    tconf[b, best_n, gj, gi] = 1
