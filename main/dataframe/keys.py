def get_keys(class_labels):
    """
    Obtain the keys of the class labels when accessing the class labels from a Pandas DataFrame.
    
    args:
    class_labels (str): the class labels to consider.
    
    returns:
    keys (list): the keys of the class labels
    """
    match class_labels:
        case 'binary':
            keys = [
            'smooth-or-featured-gz2_smooth_fraction',
            ]
            return keys
    
        case'Q0':
            keys = [
                'smooth-or-featured-gz2_smooth_fraction',
                'smooth-or-featured-gz2_featured-or-disk_fraction',
                'smooth-or-featured-gz2_artifact_fraction',
            ] 
            return keys
    
        case 'all':
            keys = [
                # Q0/ Smooth or Featured?
                'smooth-or-featured-gz2_smooth_fraction',
                'smooth-or-featured-gz2_featured-or-disk_fraction',
                'smooth-or-featured-gz2_artifact_fraction',
                
                # Q1/ Disk Edge On?
                'disk-edge-on-gz2_yes_fraction',
                'disk-edge-on-gz2_no_fraction',
                
                # Q2/ Has spiral arms?
                'has-spiral-arms-gz2_yes_fraction',
                'has-spiral-arms-gz2_no_fraction',
                
                # Q3/ Bar?
                'bar-gz2_yes_fraction',
                'bar-gz2_no_fraction',
                
                # Q4/ Bulge size?
                'bulge-size-gz2_dominant_fraction',
                'bulge-size-gz2_obvious_fraction',
                'bulge-size-gz2_just-noticeable_fraction',
                'bulge-size-gz2_no_fraction', # this is the same as 'bulge-size-gz2_none_fraction', should probably combine these into a single yes/no label pair.
                
                # Q5/ Something odd?
                'something-odd-gz2_yes_fraction',
                'something-odd-gz2_no_fraction',
                
                # Q6/ Round?
                'how-rounded-gz2_round_fraction',
                'how-rounded-gz2_in-between_fraction',
                'how-rounded-gz2_cigar_fraction',
                
                # Q7/ Bulge shape?
                'bulge-shape-gz2_round_fraction',
                'bulge-shape-gz2_boxy_fraction',
                'bulge-shape-gz2_no-bulge_fraction',
                
                # Q8/ Spiral winding?
                'spiral-winding-gz2_tight_fraction',
                'spiral-winding-gz2_medium_fraction',
                'spiral-winding-gz2_loose_fraction',
                
                # Q9/ Spiral arms count?
                'spiral-arm-count-gz2_1_fraction',
                'spiral-arm-count-gz2_2_fraction',
                'spiral-arm-count-gz2_3_fraction',
                'spiral-arm-count-gz2_4_fraction',
                'spiral-arm-count-gz2_more-than-4_fraction',
                'spiral-arm-count-gz2_cant-tell_fraction',
            ]
            return keys
        
        case 'reduced':
            keys = [
                # Q0/ Smooth or Featured?
                'smooth-or-featured-gz2_smooth_fraction',
                'smooth-or-featured-gz2_featured-or-disk_fraction',
                'smooth-or-featured-gz2_artifact_fraction',
                
                # Q1/ Disk Edge On?
                'disk-edge-on-gz2_yes_fraction',
                'disk-edge-on-gz2_no_fraction',
                
                # Q2/ Has spiral arms?
                'has-spiral-arms-gz2_yes_fraction',
                'has-spiral-arms-gz2_no_fraction',
                
                # Q3/ Bar?
                'bar-gz2_yes_fraction',
                'bar-gz2_no_fraction',
            ]
            return keys
        case _:
            raise ValueError(f"Invalid representation parameter provided: {class_labels}")

def class_labels_to_question():
    """
    Obtain the question number associated with each class label.

    returns:
    question_dict (dict): the question number associated with each class label
    """
    
    question_dict = {
            # Q0/ Smooth or Featured?
            'smooth-or-featured-gz2_smooth_fraction': 0,
            'smooth-or-featured-gz2_featured-or-disk_fraction': 0,
            'smooth-or-featured-gz2_artifact_fraction': 0,
            
            # Q1/ Disk Edge On?
            'disk-edge-on-gz2_yes_fraction': 1,
            'disk-edge-on-gz2_no_fraction': 1,
            
            # Q2/ Has spiral arms?
            'has-spiral-arms-gz2_yes_fraction': 2,
            'has-spiral-arms-gz2_no_fraction': 2,
            
            # Q3/ Bar?
            'bar-gz2_yes_fraction': 3,
            'bar-gz2_no_fraction': 3,
            
            # Q4/ Bulge size?
            'bulge-size-gz2_dominant_fraction': 4,
            'bulge-size-gz2_obvious_fraction': 4,
            'bulge-size-gz2_just-noticeable_fraction': 4,
            'bulge-size-gz2_no_fraction': 4,
            
            # Q5/ Something odd?
            'something-odd-gz2_yes_fraction': 5,
            'something-odd-gz2_no_fraction': 5,
            
            # Q6/ Round?
            'how-rounded-gz2_round_fraction': 6,
            'how-rounded-gz2_in-between_fraction': 6,
            'how-rounded-gz2_cigar_fraction': 6,
            
            # Q7/ Bulge shape?
            'bulge-shape-gz2_round_fraction': 7,
            'bulge-shape-gz2_boxy_fraction': 7,
            'bulge-shape-gz2_no-bulge_fraction': 7,
            
            # Q8/ Spiral winding?
            'spiral-winding-gz2_tight_fraction': 8,
            'spiral-winding-gz2_medium_fraction': 8,
            'spiral-winding-gz2_loose_fraction': 8,
            
            # Q9/ Spiral arms count?
            'spiral-arm-count-gz2_1_fraction': 9,
            'spiral-arm-count-gz2_2_fraction': 9,
            'spiral-arm-count-gz2_3_fraction': 9,
            'spiral-arm-count-gz2_4_fraction': 9,
            'spiral-arm-count-gz2_more-than-4_fraction': 9,
            'spiral-arm-count-gz2_cant-tell_fraction': 9,
    }
    
    return question_dict