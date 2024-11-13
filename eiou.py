
def box_area(left, upper, right, lower):
    return (right - left) * (lower - upper)

def intersection_area(box_a, box_b):
    left_intersect = max(box_a['left'], box_b['left'])
    upper_intersect = max(box_a['upper'], box_b['upper'])
    right_intersect = min(box_a['right'], box_b['right'])
    lower_intersect = min(box_a['lower'], box_b['lower'])
    
    if right_intersect > left_intersect and lower_intersect > upper_intersect:
        return (right_intersect - left_intersect) * (lower_intersect - upper_intersect)
    else:
        return 0.0

def edge_iou(box_a, box_b):
    area_a = box_area(box_a['left'], box_a['upper'], box_a['right'], box_a['lower'])
    area_b = box_area(box_b['left'], box_b['upper'], box_b['right'], box_b['lower'])
    intersect_area = intersection_area(box_a, box_b)
    iou = intersect_area / (area_a + area_b - intersect_area)
    center_x_a = (box_a['left'] + box_a['right']) / 2
    center_y_a = (box_a['upper'] + box_a['lower']) / 2
    center_x_b = (box_b['left'] + box_b['right']) / 2
    center_y_b = (box_b['upper'] + box_b['lower']) / 2
    
    distance_centers = ((center_x_a - center_x_b) ** 2 + (center_y_a - center_y_b) ** 2) ** 0.5
    max_dist = ((max(box_a['right'], box_b['right']) - min(box_a['left'], box_b['left'])) ** 2 +
                (max(box_a['lower'], box_b['lower']) - min(box_a['upper'], box_b['upper'])) ** 2) ** 0.5
    
    if max_dist == 0:
        distance_term = 0
    else:
        distance_term = distance_centers / max_dist
    
    eiou = iou - distance_term
    return eiou

def nms_with_eiou(boxes, threshold=0.5):
    if len(boxes) == 0:
        return []
    boxes.sort(key=lambda x: x['prob'], reverse=True)
    
    keep = []
    while boxes:
        largest_box = boxes.pop(0)
        keep.append(largest_box)
        remaining_boxes = []
        for box in boxes:
            eiou = edge_iou(largest_box, box)
            if eiou < threshold:
                remaining_boxes.append(box)
        
        boxes = remaining_boxes
    
    return keep

def resultnew(result,threshold=0.25):
    result0 = nms_with_eiou(result, threshold=0.25)
    return result0