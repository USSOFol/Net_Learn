"""
说明：
本代码是对深度学习中，目标检测锚框的一系列函数的实现
"""
"""
对于目标检测进行画框处理
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2


class Box:
    """

    """
    """定义两种画框的方法"""
    @staticmethod
    def box_corner_to_center(boxes):
        """
        :param boxes:
        :return: (中心点横坐标，中心点纵坐标，宽度，高度)
        """
        """从（左上，右下）转换到（中间，宽度，高度）"""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        boxes = torch.stack((cx, cy, w, h), axis=-1)
        #定义对角线画框
        return boxes

    @staticmethod
    def box_center_to_corner(boxes):
        #定义中心点画框
        """从（中间，宽度，高度）转换到（左上，右下）"""
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        boxes = torch.stack((x1, y1, x2, y2), axis=-1)
        return boxes

    @staticmethod
    def rect_(cat_box,edgcolor,line_width):
        """
        :param cat_box: [x1,y1,x2,y2]
        :param edgcolor: 'red','blue'
        :return: rect
        """
        rect = plt.Rectangle(xy=(cat_box[0], cat_box[1]),
                             width=cat_box[2] - cat_box[0],
                             height=cat_box[3] - cat_box[1],
                             fill=False,
                             edgecolor=edgcolor,
                             linewidth=line_width)
        return rect
    @staticmethod
    def show_boxes(axes, bboxes, labels=None, colors=None):
        """
        :param axes: figure的索引
        :param bboxes: 锚框坐标点，输入格式一定为(m,4)
        :param labels: 标签
        :param colors: 颜色
        :return: 返回值为带有锚框的图，若要显示，须在本函数后添加plt.show()
        """
        """显示所有边界框"""

        def _make_list(obj, default_values=None):
            if obj is None:
                obj = default_values
            elif not isinstance(obj, (list, tuple)):
                obj = [obj]
            return obj

        labels = _make_list(labels)
        # 转为列表
        #print('labels',labels)
        colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
        # print('colors',colors)
        for i, bbox in enumerate(bboxes):
            color = colors[i % len(colors)]
            rect = Box.rect_(bbox.detach().numpy(), edgcolor=color, line_width=3)
            axes.add_patch(rect)
            if labels and len(labels) > i:
                text_color = 'k' if color == 'w' else 'w'
                axes.text(rect.xy[0], rect.xy[1], labels[i],
                          va='center', ha='center', fontsize=9, color=text_color,
                          bbox=dict(facecolor=color, lw=0))
    """锚框法"""
    @staticmethod
    def multibox_prior(data,sizes,ratios):
        """
        Y = Box.multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
        :param data: 输入的一个随机阵列
        :param sizes: 缩放比
        :param ratios:宽高比
        :return: 返回值为锚框的对角坐标的归一化，[x1,y1,x2,y2]
        需要×[w,h,w,h]回归坐标
        由于一切计算都要归于深度学习的计算，因此这里的返回值在0-1之间是有原因的
        """
        in_height,in_width = data.shape[-2:]
        # 获取高与宽
        # print('in_height',in_height)
        # print(in_width)
        device,num_sizes,num_ratios = data.device,len(sizes),len(ratios)
        #print(device,num_sizes,num_ratios)
        # 一些相关参数
        boxes_per_pixel = (num_sizes+num_ratios-1)#生成的锚定框的个数
        size_tensor = torch.tensor(sizes,device=device)
        # 缩放比的张量
        ratios_tensor = torch.tensor(ratios,device = device)
        # 宽高比的张量

        ###############设置偏移量####################
        offset_h, offset_w = 0.5, 0.5
        #
        step_h = 1.0/in_height # 在y轴上的缩放步长
        step_w = 1.0/in_width  # 在x轴上的缩放步长
        ## 什么是缩放步长
        #print(step_h,step_w)

        ###############生成锚框的所有中心点###########################
        center_h = (torch.arange(in_height,device = device) + offset_h) * step_h
        center_w = (torch.arange(in_width, device=device) + offset_w) * step_w
        #print(center_h)
        #print(center_w)
        shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
        #print(shift_x.size(),shift_x.size())
        shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)
        ##这里进行整形
        #print(shift_x.size(), shift_x.size())

        ##########################################################
        # 生成所有锚定框的长和宽
        # 创建四角坐标，即对角坐标
        w = torch.cat((size_tensor * torch.sqrt(ratios_tensor[0]),
                       sizes[0] * torch.sqrt(ratios_tensor[1:]))) * in_height / in_width  # 处理矩形输入
        #print("w",w)
        h = torch.cat((size_tensor / torch.sqrt(ratios_tensor[0]),
                       sizes[0] / torch.sqrt(ratios_tensor[1:])))
        #print("h",h)
        ## 除以2获得半高宽
        anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
            in_height * in_width, 1) / 2
        #print("anchor_manipulations",anchor_manipulations)
        # 每个中心点都将有“boxes_per_pixel”个锚框，
        # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
        out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                               dim=1).repeat_interleave(boxes_per_pixel, dim=0)
        #print("out_grid",out_grid)
        output = out_grid + anchor_manipulations
        # 这里输出的并不是原始坐标值，而是一个相对比值，这个比值乘以对应的原图的wh即可还原
        return output.unsqueeze(0)
    @staticmethod
    def iou(boxes1,boxes2):
        """
        :param boxes1:锚框1
        :param boxes2: 锚框2
        :return: 交并比
        """
        #intersection over union 交并比，指(相交面积)/(并面积),如果二者重合度越高越接近于1，否则越趋近于0
        box_area = lambda boxes:(
            (boxes[:, 2] - boxes[:, 0]) *
            (boxes[:, 3] - boxes[:, 1])
        )
        areas1 = box_area(boxes1)
        areas2 = box_area(boxes2)
        # inter_upperlefts,inter_lowerrights,inters的形状:
        # (boxes1的数量,boxes2的数量,2)
        inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
        # inter_areasandunion_areas的形状:(boxes1的数量,boxes2的数量)
        inter_areas = inters[:, :, 0] * inters[:, :, 1]
        union_areas = areas1[:, None] + areas2 - inter_areas
        return inter_areas / union_areas

    @staticmethod
    def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
        """
        将最接近的真实边界框分配给锚框"""
        """
        anchors_bbox_map = Box.assign_anchor_to_bbox(label[:, 1:], anchors, device)
        """
        # 输入真实框的比例尺，锚框位置，device,这里默认iou的阈值为0.5
        num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
        # 框的数量，真实的框的数量
        # 对于深度学习，一次输入不止一组框，这里先获得几组训练样本
        # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
        jaccard = Box.iou(anchors, ground_truth)
        # print("jaccard",jaccard)
        # 对于每个锚框，分配的真实边界框的张量
        # 锚框与真实框的iou
        anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                      device=device)
        # 根据阈值，决定是否分配真实边界框
        max_ious, indices = torch.max(jaccard, dim=1)
        # 返回最大iou值与其索引,按照行进行挑选最大值
        # print("max_iou",max_ious)
        # print("indics",indices)
        anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
        # 给出行数
        # 选出其中不是零的大于索引值得给出索引
        # print("anc_i",anc_i)
        box_j = indices[max_ious >= iou_threshold]
        # 给出列数
        # print("box_j",box_j)
        anchors_bbox_map[anc_i] = box_j
        # print('anchor_map',anchors_bbox_map)
        col_discard = torch.full((num_anchors,), -1)
        row_discard = torch.full((num_gt_boxes,), -1)
        # num_get_boxes:指一个图片里面真实位置有几个
        """
        算法详解：
        前面已经获得了一个锚框相对于真实框的一个uoi矩阵，那么现在进行锚框最相关匹配
        但是，注意，但是为了防止这个最佳匹配直接删掉所有的备选锚框，必须先根据阈值进行一次筛选，
        根据阈值筛选后才能进一步的进行筛选，两次筛选加补充才能最好的区分背景和目标
        首先将最大的那一个uoi选择出来，然后根据坐标认定为映射，即最相关，将这一行这一列全部赋值为-1
        接着剩下的进行寻找，然后映射，然后继续赋值为-1
        这一部分分为两部：1.阈值映射获取。2.最相关获取
        """
        for _ in range(num_gt_boxes):
            max_idx = torch.argmax(jaccard)
            # 给出全局uoi最大的索引
            box_idx = (max_idx % num_gt_boxes).long()
            # print("box_idx",box_idx)
            #
            anc_idx = (max_idx / num_gt_boxes).long()
            #
            # print(anc_i)
            anchors_bbox_map[anc_idx] = box_idx
            #################################
            jaccard[:, box_idx] = col_discard
            jaccard[anc_idx, :] = row_discard
        return anchors_bbox_map

    @staticmethod
    def offset_boxes(anchors, assigned_bb, eps=1e-6):
        """对锚框偏移量的转换,注意，这个偏移量就是神经网络最后输出的东西"""
        c_anc = Box.box_corner_to_center(anchors)
        # 锚框
        c_assigned_bb = Box.box_corner_to_center(assigned_bb)
        # 真实
        offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
        offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
        offset = torch.cat([offset_xy, offset_wh], axis=1)
        return offset

    @staticmethod
    def offset_inverse(anchors, offset_preds):
        """
        :param anchors: 锚框
        :param offset_preds: 偏移量
        :return: 目标框对角坐标点
        """
        """根据带有预测偏移量的锚框来预测边界框，对预测的偏移量的逆运算就是目标框"""
        anc = Box.box_corner_to_center(anchors)
        pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
        pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
        pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
        predicted_bbox = Box.box_center_to_corner(pred_bbox)
        return predicted_bbox

    @staticmethod
    def multibox_target(anchors, labels):
        """
        :param anchors:
        :param labels:
        :return: （每个锚框的偏移值（背景=0），背景和锚框的标志，分类）
        """
        """使用真实边界框标记锚框"""
        """
        labels = Box.multibox_target(anchors.unsqueeze(dim=0),
                             ground_truth.unsqueeze(dim=0))
        ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                                    [1, 0.55, 0.2, 0.9, 0.88]])
        # 真实框位置
        anchors = torch.tensor([[0, 0.1, 0.2, 0.3],
                                [0.15, 0.2, 0.4, 0.4],
                                [0.63, 0.05, 0.88, 0.98],
                                [0.66, 0.45, 0.8, 0.8],
                                [0.57, 0.3, 0.92, 0.9]])
        """
        batch_size, anchors = labels.shape[0], anchors.squeeze(0)
        #
        batch_offset, batch_mask, batch_class_labels = [], [], []
        # 设置空列表
        device, num_anchors = anchors.device, anchors.shape[0]
        for i in range(batch_size):
            label = labels[i, :, :]
            # print("labels",label)
            anchors_bbox_map = Box.assign_anchor_to_bbox(
                label[:, 1:], anchors, device)
            # 输入真实框的比例尺，锚框位置，device
            # anco_box_map tensor([-1,  0,  1, -1,  1])
            #print("anco_box_map",anchors_bbox_map)
            """将最接近的真实边界框分配给锚框"""
            bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
                1, 4)
            #print("mask",bbox_mask)
            # 将背景变为0，之后用掩膜相乘进行去除
            # 将类标签和分配的边界框坐标初始化为零
            class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                       device=device)
            assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                      device=device)
            # 使用真实边界框来标记锚框的类别。
            # 如果一个锚框没有被分配，我们标记其为背景（值为零）
            indices_true = torch.nonzero(anchors_bbox_map >= 0)
            bb_idx = anchors_bbox_map[indices_true]
            class_labels[indices_true] = label[bb_idx, 0].long() + 1
            assigned_bb[indices_true] = label[bb_idx, 1:]
            # 偏移量转换
            offset = Box.offset_boxes(anchors, assigned_bb) * bbox_mask
            batch_offset.append(offset.reshape(-1))
            batch_mask.append(bbox_mask.reshape(-1))
            batch_class_labels.append(class_labels)
        """常见拼接函数有cat stack两种"""
        #print(batch_mask)
        bbox_offset = torch.stack(batch_offset)
        bbox_mask = torch.stack(batch_mask)
        #print(bbox_mask)
        class_labels = torch.stack(batch_class_labels)
        return (bbox_offset, bbox_mask, class_labels)
    #####################################################
    """
    非极大值抑制：
    对于一个图中的某一类别进行识别之时，假设一张图片中存在两条狗，一只猫，那么会出现下面的情况：
    在生成了大量锚框之后。经过前面的一系列操作，删除了背景框之后，那么围绕这三个目标会有下列情况
    围绕狗1有n个锚框
    围绕狗2有m个锚框
    围绕猫有j个锚框
    这些锚框内的物体经由网络识别后会得到一个物体概率，每个锚框将会得到一个概率，那么有下面的任务要求：
    1.保留每个物体的最高置信度锚框
    2.防止狗1，狗2的锚框之间出现仅仅保留一个的情况
    对于只有一只猫猫的情况好办，直接删除类别猫除最高概率以下的所有锚框
    对于狗子，第一，找出概率最大的锚框，
    第二，计算其他所有类别为狗的锚框与这个锚框的交并比，根据阈值进行删除，认为第一条狗的锚框已确定
    第三，对剩余锚框重复上诉操作，直到没有锚框剩下
    """
    @staticmethod
    def nms(boxes, scores, iou_threshold):
        """
        :param boxes:输入全体锚框
        :param scores: 每个锚框经由神经网络的测量
        :param iou_threshold: iou判别阈值
        :return: 保留的锚框
        对预测边界框的置信度进行排序
        第一步：获取评分最大值框坐标
        第二步：计算其余与其的uoi
        第三步：删除高于阈值的/保留低的
        第四步：找第二个高的循环"""
        B = torch.argsort(scores, dim=-1, descending=True)
        #print('B',B)
        # B是将每一行最大那个置为0
        keep = []  # 保留预测边界框的指标
        while B.numel() > 0:
            i = B[0]
            # 输出每一轮B的第一行
            keep.append(i)
            if B.numel() == 1:
                #如果就一个评分，直接关了吧，没意思
                break
            iou = Box.iou(boxes[i, :].reshape(-1, 4),
                        boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
            #print('iou',iou)
            inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
            # 将大于阈值的全部归零，返回的数字中不为0的就是原来iou中小于阈值的索引值
            #print('inds:', inds)
            B = B[inds + 1]
            #print('B:',B)
        return torch.tensor(keep, device=boxes.device)

    @staticmethod
    def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                               pos_threshold=0.009999999):
        """
        :param cls_probs: 每一个锚框对于其分类的类别的概率，若分类四种，有六个框，则为 [batch_sizex4x6] 的矩阵
        :param offset_preds: 预测的锚框的偏移值
        :param anchors: 锚框对角坐标
        :param nms_threshold: 判定是否为同一对象的不同框的交并比阈值
        :param pos_threshold:
        :return:  [分类代号(-1删除代号) , 概率 ，锚框坐标]
        """
        """使用非极大值抑制来预测边界框"""
        device, batch_size = cls_probs.device, cls_probs.shape[0]
        anchors = anchors.squeeze(0)
        num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
        # 几个类别，几个锚框
        out = []
        # 输出的预留空列表
        for i in range(batch_size):
            cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)

            conf, class_id = torch.max(cls_prob[1:], 0)
            # 给出非背景的排序，同时拿出每一列的最大值,给出最大值是在哪一行拿的
            #print('cls_prob',cls_prob[1:])
            #print('conf',conf)
            #print(class_id)
            # 找出每一行中的最大值
            predicted_bb = Box.offset_inverse(anchors, offset_pred)
            # 反算预测对角点
            #print('pre_bb',predicted_bb)
            keep = Box.nms(predicted_bb, conf, nms_threshold)
            #print('keep',keep)
            # 找到所有的non_keep索引，并将类设置为背景
            all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
            combined = torch.cat((keep, all_idx))
            #print('combine',combined)
            uniques, counts = combined.unique(return_counts=True)
            #print(uniques,counts)
            non_keep = uniques[counts == 1]
            #print('nonekeep',non_keep)
            all_id_sorted = torch.cat((keep, non_keep))
            class_id[non_keep] = -1
            class_id = class_id[all_id_sorted]
            conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
            # pos_threshold是一个用于非背景预测的阈值
            below_min_idx = (conf < pos_threshold)
            class_id[below_min_idx] = -1
            conf[below_min_idx] = 1 - conf[below_min_idx]
            pred_info = torch.cat((class_id.unsqueeze(1),
                                   conf.unsqueeze(1),
                                   predicted_bb), dim=1)
            out.append(pred_info)
        return torch.stack(out)
    ###################################################
    """
    上述中都是根据每个像素点进行一个锚框的采样生成，这样的话图就太多了，不好检测，
    那么在图像上进行均匀采样可以降低计算量
    那么我们就像定义这些量：
    1.高方向上采样中心点的数量，比如均匀间隔采四个
    2.宽方向上的采样中心点数量，比如均匀采样采集五个
    3.采样框的高宽比r，以及缩放率w，以确定每个采样点生成的采样框个数
    """
    @staticmethod
    def display_anchors(Batch_size,deep,img,f_w,f_h,s,r):
        """
        :param img:输入图
        :param f_w: 宽方向上的采样个数
        :param f_h: 高方向上的采样个数
        :param s: 锚框比例
        :param r: 高宽比
        :return: [m,4]的锚框对角坐标
        """
        f_map = torch.zeros((Batch_size,deep,f_h,f_w))
        anchors = Box.multibox_prior(f_map,s,r)
        scales = torch.tensor(img.shape[:2]).repeat(1,2).fliplr().squeeze(0)
        return (anchors*scales).view(-1,4)



if __name__ == "__main__":
    ###########################################
    img = cv2.imread("my_cat.jpg")
    #img = transforms.ToPILImage(img)#用于transforms方法使用
    plt.figure()
    fig = plt.imshow(img)
    ###实现功能：给出框坐标给框出来
    # bbox是边界框的英文缩写
    cat_box = [750.0,100,1500,800]
    boxes = torch.tensor([cat_box])
    # print(boxes.size())
    # print(Box.box_center_to_corner(Box.box_corner_to_center(boxes)) == boxes)
    rect = Box.rect_(cat_box,'purple',3)
    fig.axes.text(750+10, 100+50, "my_cat 0.9", bbox={'facecolor': 'blue', 'alpha': 0.5})
    # 添加标签
    fig.axes.add_patch(rect)
    # 添加框框
    plt.show()

##########################################################
    h ,w= img.shape[:2]
    #img.shape[]=[hxwxdeep]
    # print(h,w)
    # 输出高和宽
    X = torch.rand(size=(1, 3, h, w))
    # 这个随机数组干啥的范围是0-1
    # print(X)
    Y = Box.multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    boxes = Y.reshape(h, w, 5, 4)
    print('boxes1:',boxes)
    # 每一个中心点都对应着5个锚定框，每个锚定框有四个点
    # print(boxes[250, 250, 0, :])
    boxes_fig = plt.imshow(img)
    wh = torch.tensor((w, h, w, h))
    i = ['red', 'blue', 'yellow', 'brown', 'purple']
    for x,y in enumerate(boxes[400,1100,:,:]):
        box = Box.rect_(y*wh,i[x],3)
        boxes_fig.axes.add_patch(box)
    plt.show()
    ##############################################
    ## 利用交并比来实现最优框的选取
    ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                                 [1, 0.55, 0.2, 0.9, 0.88]])
    # 真是框位置
    anchors = torch.tensor([[0, 0.1, 0.2, 0.3],
                            [0.15, 0.2, 0.4, 0.4],
                            [0.63, 0.05, 0.88, 0.98],
                            [0.66, 0.45, 0.8, 0.8],
                            [0.57, 0.3, 0.92, 0.9]])
    # anchors 位置
    fig = plt.imshow(img)
    print(anchors * wh)
    Box.show_boxes(fig.axes, anchors * wh,
                ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
    Box.show_boxes(fig.axes,ground_truth[:,1:]*wh,['cat1',"cat2"],'k')
    plt.show()
    labels = Box.multibox_target(anchors.unsqueeze(dim=0),
                             ground_truth.unsqueeze(dim=0))
    print('labels:',labels)
    ##############################################################
    """使用非极大抑制"""
    anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92],
                            [0.08, 0.2, 0.56, 0.95],
                            [0.15, 0.3, 0.62, 0.91],
                            [0.55, 0.2, 0.9, 0.88]])
    offset_preds = torch.tensor([0] * anchors.numel())
    print(offset_preds)
    cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
                              [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                              [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
    output = Box.multibox_detection(cls_probs.unsqueeze(dim=0),
                                offset_preds.unsqueeze(dim=0),
                                anchors.unsqueeze(dim=0),
                                nms_threshold=0.5)
    # 注意，这里升维的原因是用于深度学习的时候，要加个batch_size
    print(output)
    #####################################
    """均匀采样获取锚框"""
    boxes = Box.display_anchors(Batch_size=1,deep = 2,img=img,f_w=3,f_h=3,s = [0.15,0.12],r =[1,2,0.5])
    print(boxes)
    fig1 = plt.imshow(img)
    Box.show_boxes(fig1.axes, boxes )
    plt.show()




