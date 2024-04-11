# Galaxy Zoo 2 dataset analysis

"""
167434 images of distinct galaxies in the dataset

class labels are a number from 0 to 6, each number represents a different class, with -1 being an unclassified class
class names: 7 distinct classes. 1 unclassified class. 

-----------------------------------------------------------------------------------------------------------
| Class Label |                   Class Name                       |  Number of Galaxies  | % of Galaxies |    
|-------------|----------------------------------------------------|----------------------|---------------|                        
|    -1       | Missing Value (could be unclassified or artifact)  |    9461 galaxies     |     5.65%     |
|     0       | "smooth_inbetween"                                 |    62857 galaxies    |     37.54%    |
|     1       | "smooth_round"                                     |    43800 galaxies    |     26.16%    |
|     2       | "smooth_cigar"                                     |    15801 galaxies    |     9.44%     |
|     3       | "edge_on_disk"                                     |    3729 galaxies     |     2.23%     |
|     4       | "unbarred_spiral"                                  |    20992 galaxies    |     12.54%    |
|     5       | "barred_spiral"                                    |    7346 galaxies     |     4.39%     |
|     6       | "featured_without_bar_or_spiral"                   |    3448 galaxies     |     2.06%     |
-----------------------------------------------------------------------------------------------------------

can interpret these as the basic classes of galaxies. coarse features of galaxies that are easy to classify.
the other features are more fine grained, and are not as easy to classify as they are rarer. 

Galaxy Zoo 2 decision tree:

    1st level of decision tree:
    T00: Is the galaxy simply smooth and rounded, with no sign of a disk?
        A0: smooth, A1: features/disk, A2: star/artifact

    T05: Is thre anything odd?
        A0: yes, A1: no

    2nd level of decision tree:
        T07: How rounded is the galaxy?
            A0: completely round, A1: in between, A2: cigar shaped

        T01: Could this be a disk viewed edge-on?
            A0: yes, A1: no

        3rd level of decision tree:
            T08: does the galaxy have a bulge at the centre? if so, what is the shape?
                A0: rounded, A1: boxy, A2: none

            T02: Is there any sign of a bar feature through the centre of the galaxy?
                A0: yes, A1: no

            T03: Is there any sign of spiral arms in the galaxy?
                A0: yes, A1: no

            T04: How prominent is the central bulge, compared with the rest of the galaxy?
                A0: no bulge, A1: just noticable, A2: obvious, A3: dominant

            T06: Is the odd feature a ring, or is the galaxy disturbed or irregular?
                A0: ring, A1: lens or arc, A2: disturbed, A3: irregular, A4: other, A5: merger, A6: dust lane

            4th level of decision tree:
                T09: How tightly wound are the spiral arms?
                    A0: tight, A1: medium, A2: loose

                T10: How many spiral arms are there?
                    A0: 1, A1: 2, A2: 3, A3: 4, A4: more than 4, A5: can't tell

37 Options in total in the decision tree

Percentages of answers to the questions in the decision tree: (data from datawrangler, need to go through this manually do some more analysis and get information on some of the questions)

75% of the galaxies are classified as smooth
24% of the galaxies are classified as featured-or-disk
<1% of the galaxies are classified as deadlock
<1% of the galaxies are classified as artifact

76% of the galaxies are unclassified in the disk edge on question
24% of the galaxies are classified as no
3% of the galaxies are classified as yes
<1% of the galaxies are classified as deadlock

76% of the galaxies are unclassified in the has spiral arms question
18% of the galaxies are classified as yes
4% of the galaxies are classified as no
2% of the galaxies are classified as deadlock

76% of the galaxies are unclassified in the bar question
6% of the galaxies are classified as yes
16% of the galaxies are classified as no
2% of the galaxies are classified as deadlock

76% of the galaxies are unclassified in the bulge size question
13% of the galaxies are classified as just noticable
7% of the galaxies are classified as obvious
3% of the galaxies are classified as dominant or deadlock

91% of the galaxies do not have something odd
8% of the galaxies do have something odd
<1% of the galaxies are deadlock

26% of the galaxies are unclassified in the how rounded question
26% of the galaxies are classified as round
38% of the galaxies are classified as in-between
10% of the galaxies are classified as cigar or deadlock

97% of the galaxies are unclassified in the bulge shape question
2% of the galaxies are classified as round
1% have no bulge
<1% are classified as boxy

87% of the galaxies are unclassified in the spiral winding question
6% of the galaxies are classified as medium
4% of the galaxies are classified as tight
2% of the galaxies are classified as loose

87% of the galaxies are unclassified in the spiral arm count question
8% of the galaxies are classified as 2
3% of the galaxies are classified as can't tell
2% of the galaxies are classified as 3, 4, or more than 4

38% of the galaxies are smooth inbetween
26% of the galaxies are smooth round
13% of the galaxies are unbared spiral
18% of the galaxies are "other"

NOTE: percentages obtained from datawrangler, I need to go through this manually do some more analysis

There are 90 columns in the dataset with keys and values (of vectors) corresponding to:

'ra' = right ascension
'dec' = declination
'smooth-or-featured-gz2_smooth' = number of votes for smooth
'smooth-or-featured-gz2_featured-or-disk' = number of votes for featured or disk
'smooth-or-featured-gz2_artifact' = number of votes for artifact
'disk-edge-on-gz2_yes' = number of votes for seen edge on
'disk-edge-on-gz2_no' = number of votes for not seen edge on
'bar-gz2_yes' = number of votes for bar
'bar-gz2_no' = number of votes for no bar
'has-spiral-arms-gz2_yes' = number of votes for spiral arms
'has-spiral-arms-gz2_no' = number of votes for no spiral arms
'bulge-size-gz2_no' = number of votes for no bulge
'bulge-size-gz2_just-noticeable' = number of votes for just noticeable bulge
'bulge-size-gz2_obvious' = number of votes for obvious bulge
'bulge-size-gz2_dominant' = number of votes for dominant bulge
'something-odd-gz2_yes' = number of votes for something odd in the image
'something-odd-gz2_no' = number of votes for nothing odd in the image
'how-rounded-gz2_round' = number of votes for round galaxy shape
'how-rounded-gz2_in-between'= number of votes for inbetween galaxy shape
'how-rounded-gz2_cigar' = number of votes for cigar galaxy shape
'bulge-shape-gz2_round' = number of votes for round bulge shape
'bulge-shape-gz2_boxy' = number of votes for boxy bulge shape
'bulge-shape-gz2_no-bulge' = number of votes for no bulge
'spiral-winding-gz2_tight' = number of votes for tight spiral arm winding
'spiral-winding-gz2_medium' = number of votes for medium spiral arm winding
'spiral-winding-gz2_loose' = number of votes for loose spiral arm winding
'spiral-arm-count-gz2_1' = number of votes for 1 spiral arm
'spiral-arm-count-gz2_2' = number of votes for 2 spiral arms
'spiral-arm-count-gz2_3' = number of votes for 3 spiral arms
'spiral-arm-count-gz2_4' = number of votes for 4 spiral arms
'spiral-arm-count-gz2_more-than-4' = number of votes for more than 4 spiral arms
'spiral-arm-count-gz2_cant-tell' = number of votes for can't tell the no. of spiral arms
'smooth-or-featured-gz2_total-votes' = total number of votes to the smooth or featured question
'smooth-or-featured-gz2_smooth_fraction' = probability that the galaxy in the image is smooth
'smooth-or-featured-gz2_featured-or-disk_fraction'= probability that the galaxy in the image is featured or a disk
'smooth-or-featured-gz2_artifact_fraction'= probability that the galaxy in the image is an artifact
'disk-edge-on-gz2_total-votes' = total number of votes to the disk edge on question
'disk-edge-on-gz2_yes_fraction' = probability that the galaxy in the image is seen edge on
'disk-edge-on-gz2_no_fraction' = probability that the galaxy in the image is not seen edge on
'has-spiral-arms-gz2_total-votes' = total number of votes to the has spiral arms question
'has-spiral-arms-gz2_yes_fraction' = probability that the galaxy in the image has spiral arms
'has-spiral-arms-gz2_no_fraction' = probability that the galaxy in the image has no spiral arms
'bar-gz2_total-votes' = total number of votes to the bar question
'bar-gz2_yes_fraction' = probability that the galaxy in the image has a bar
'bar-gz2_no_fraction' = probability that the galaxy in the image doesn't have bar
'bulge-size-gz2_total-votes' = total number of votes to the bulge size question
'bulge-size-gz2_dominant_fraction' = probability that the galaxy in the image has a dominant bulge
'bulge-size-gz2_obvious_fraction' = probability that the galaxy in the image has an obvious bulge
'bulge-size-gz2_just-noticeable_fraction' = probability that the galaxy in the image has a just noticeable bulge
'bulge-size-gz2_no_fraction' = probability that the galaxy in the image has no bulge
'something-odd-gz2_total-votes' = total number of votes for the something odd question
'something-odd-gz2_yes_fraction' = probability that the galaxy in the image has something odd in it
'something-odd-gz2_no_fraction' = probability that the galaxy in the image doesn't have something odd in it
'how-rounded-gz2_total-votes' = total number of votes to the how rounded question
'how-rounded-gz2_round_fraction' = probability that the galaxy in the image is round in shape
'how-rounded-gz2_in-between_fraction' = probability that the galaxy in the image is in between in shape
'how-rounded-gz2_cigar_fraction' = probability that the galaxy in the image is cigar like in shape
'bulge-shape-gz2_total-votes' = total number of votes to the bulge shape question
'bulge-shape-gz2_round_fraction' = probability that the galaxy in the image has a round bulge
'bulge-shape-gz2_boxy_fraction' = probability that the galaxy in the image has a boxy bulge
'bulge-shape-gz2_no-bulge_fraction' = probability that the galaxy in the image has no bulge
'spiral-winding-gz2_total-votes' = total number of votes to the spiral winding question
'spiral-winding-gz2_tight_fraction' = probability that the galaxy in the image has tight spiral winding
'spiral-winding-gz2_medium_fraction' = probability that the galay in the image has medium spiral winding
'spiral-winding-gz2_loose_fraction' = probability that the galaxy in the image has loose spiral winding
'spiral-arm-count-gz2_total-votes' = total number of votes to the spiral arm count question
'spiral-arm-count-gz2_1_fraction' = probability that the galaxy in the image has 1 spiral arm
'spiral-arm-count-gz2_2_fraction' = probability that the galaxy in the image has 2 spiral arms
'spiral-arm-count-gz2_3_fraction' = probability that the galaxy in the image has 3 spiral arms
'spiral-arm-count-gz2_4_fraction' = probability that the galaxy in the image has 4 spiral arms
'spiral-arm-count-gz2_more-than-4_fraction' = probability that the galaxy in the image has more than 4 spiral arms
'spiral-arm-count-gz2_cant-tell_fraction' = probability that the galaxy in the image has ??? spiral arms
'file_loc' = file location of the image, it's path
'subfolder' = subfolder of the image, it's root
'iauname' = IAU Name of the galaxy
'smooth-or-featured-gz2_semantic' = final classification of the galaxy feature, either smooth or featured/disk
'disk-edge-on-gz2_semantic' = final classification of the galaxy feature, either edge on or not edge on
'has-spiral-arms-gz2_semantic' = final classifiction of the galaxy feature, either has spiral arms or doesn't
'bar-gz2_semantic' = final classification of the galaxy feature, either has a bar or doesn't
'bulge-size-gz2_semantic' = final classification of the galaxy feature, either has a just noticeable, obvious, dominant, or no bulge
'something-odd-gz2_semantic' = final classification of the galaxy feature, either has something odd or doesn't
'how-rounded-gz2_semantic' = final classification of the galaxy feature, either is rounded, inbetween, or cigar shaped
'bulge-shape-gz2_semantic' = final classification of the galaxy feature, either is round, boxy, or no bulge
'spiral-winding-gz2_semantic' = final classification of the galaxy feature, either loose, medium, or tight spiral arm winding
'spiral-arm-count-gz2_semantic' = final classification of the galaxy feature, either 1, 2, 3, 4, more than 4, or ??? number of spiral arm(s)
'summary' = final classification of the galaxy based off the final classifications of the other galaxy features.
'leaf_prob' = probability of being a leaf
'label' = final classification of the galaxy as an integer label
'filename' = the filename of the image containing the galaxy
'id_str' = the unique id given to the galaxy within the dataset
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

dataset_root = 'galaxyzoo2-dataset/'
catalog_name = r'gz2_train_catalog.parquet' 

parquet_file_path = os.path.join(dataset_root, catalog_name)

# Read the Parquet file
df = pd.read_parquet(parquet_file_path)

class_labels = df['label']

print(f"\nNumber of Galaxies in the dataset: {len(class_labels)}\n")
print(f"class labels:  {sorted(class_labels.unique())}\n")
print("Number of galaxies in each class: \n")
print(f"Artifact: {len(class_labels[class_labels == -1])} = {len(class_labels[class_labels == -1])/len(class_labels)*100:.2f} %")
print(f"Smooth Inbetween: {len(class_labels[class_labels == 0])} = {len(class_labels[class_labels == 0])/len(class_labels)*100:.2f} %")
print(f"Smooth Round: {len(class_labels[class_labels == 1])} = {len(class_labels[class_labels == 1])/len(class_labels)*100:.2f} %")
print(f"Smooth Cigar: {len(class_labels[class_labels == 2])} = {len(class_labels[class_labels == 2])/len(class_labels)*100:.2f} %")
print(f"Edge on Disk: {len(class_labels[class_labels == 3])} = {len(class_labels[class_labels == 3])/len(class_labels)*100:.2f} %")
print(f"Unbarred Spiral: {len(class_labels[class_labels == 4])} = {len(class_labels[class_labels == 4])/len(class_labels)*100:.2f} %")
print(f"Barred Spiral: {len(class_labels[class_labels == 5])} = {len(class_labels[class_labels == 5])/len(class_labels)*100:.2f} %")
print(f"Featured without bar or spiral: {len(class_labels[class_labels == 6])} = {len(class_labels[class_labels == 6])/len(class_labels)*100:.2f} %")

# plot a correlation matrix of the dataset

smooth_frac = df['smooth-or-featured-gz2_smooth_fraction']
disk_frac = df['smooth-or-featured-gz2_featured-or-disk_fraction']
artifact_frac = df['smooth-or-featured-gz2_artifact_fraction']
disk_edge_on_yes_frac = df['disk-edge-on-gz2_yes_fraction']
disk_edge_on_no_frac = df['disk-edge-on-gz2_no_fraction']
has_spiral_arms_yes_frac = df['has-spiral-arms-gz2_yes_fraction']
has_spiral_arms_no_frac = df['has-spiral-arms-gz2_no_fraction']
bar_yes_frac = df['bar-gz2_yes_fraction']
bar_no_frac = df['bar-gz2_no_fraction']
bulge_size_no_frac = df['bulge-size-gz2_no_fraction']
bulge_size_just_noticable_frac = df['bulge-size-gz2_just-noticeable_fraction']
bulge_size_obvious_frac = df['bulge-size-gz2_obvious_fraction']
bulge_size_dominant_frac = df['bulge-size-gz2_dominant_fraction']
something_odd_yes_frac = df['something-odd-gz2_yes_fraction']
something_odd_no_frac = df['something-odd-gz2_no_fraction']
how_rounded_round_frac = df['how-rounded-gz2_round_fraction']
how_rounded_in_between_frac = df['how-rounded-gz2_in-between_fraction']
how_rounded_cigar_frac = df['how-rounded-gz2_cigar_fraction']
bulge_shape_round_frac = df['bulge-shape-gz2_round_fraction']
bulge_shape_boxy_frac = df['bulge-shape-gz2_boxy_fraction']
bulge_shape_no_bulge_frac = df['bulge-shape-gz2_no-bulge_fraction']
spiral_winding_tight_frac = df['spiral-winding-gz2_tight_fraction']
spiral_winding_medium_frac = df['spiral-winding-gz2_medium_fraction']
spiral_winding_loose_frac = df['spiral-winding-gz2_loose_fraction']
spiral_arm_count_1_frac = df['spiral-arm-count-gz2_1_fraction']
spiral_arm_count_2_frac = df['spiral-arm-count-gz2_2_fraction']
spiral_arm_count_3_frac = df['spiral-arm-count-gz2_3_fraction']
spiral_arm_count_4_frac = df['spiral-arm-count-gz2_4_fraction']
spiral_arm_count_more_than_4_frac = df['spiral-arm-count-gz2_more-than-4_fraction']
spiral_arm_count_cant_tell_frac = df['spiral-arm-count-gz2_cant-tell_fraction']

data = [smooth_frac, disk_frac, artifact_frac, disk_edge_on_yes_frac, disk_edge_on_no_frac, has_spiral_arms_yes_frac, has_spiral_arms_no_frac, bar_yes_frac, bar_no_frac,
    bulge_size_no_frac, bulge_size_just_noticable_frac, bulge_size_obvious_frac, bulge_size_dominant_frac, something_odd_yes_frac, something_odd_no_frac, how_rounded_round_frac,
    how_rounded_in_between_frac, how_rounded_cigar_frac, bulge_shape_round_frac, bulge_shape_boxy_frac, bulge_shape_no_bulge_frac, spiral_winding_tight_frac, spiral_winding_medium_frac,
    spiral_winding_loose_frac, spiral_arm_count_1_frac, spiral_arm_count_2_frac, spiral_arm_count_3_frac, spiral_arm_count_4_frac, spiral_arm_count_more_than_4_frac, spiral_arm_count_cant_tell_frac]

# Remove any NaN values from the data
data = [np.nan_to_num(d) for d in data]

class_names = ['Smooth', 'Featured', 'Artifact', 'Edge on Disk', 'Not Edge on Disk', 'Bar', 'Not Bar', 'Has Spiral Arms', 'No Spiral Arms', 'No Bulge', 'Just Noticable Bulge',
        'Obvious Bulge', 'Dominant Bulge', 'Something Odd', 'Nothing Odd', 'Round', 'In between', 'Cigar', 'Round Bulge', 'Boxy Bulge', 'No Bulge', 'Tight Winding', 
        'Medium Winding', 'Loose Winding', '1 Arm', '2 Arms', '3 Arms', '4  Arms', 'More than 4 Arms', "Can't Tell no. Arms"]

questions = ['Smooth?', 'Featured?', 'Artifact?', 'Edge on Disk?', 'Not Edge on Disk?', 'Bar?', 'Not Bar?', 'Has Spiral Arms?', 'No Spiral Arms?', 
            'No Bulge?', 'Just Noticable Bulge?', 'Obvious Bulge?', 'Dominant Bulge?', 'Something Odd?', 'Nothing Odd?', 'Round?', 'In between?', 'Cigar?',
            'Round Bulge?', 'Boxy Bulge?', 'No Bulge?', 'Tight Winding?', 'Medium Winding?', 'Loose Winding?', '1 Arm?', '2 Arms?', '3 Arms?', '4  Arms?',
            'More than 4 Arms?', "Can't Tell no. Arms?"]

fig, ax = plt.subplots(1,1, figsize=(20,20), dpi = 300)

correlation_matrix = np.corrcoef(data)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, f"{correlation_matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=2)

cax = ax.matshow(correlation_matrix, cmap='PiYG')

ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(questions)))

ax.set_xticklabels(class_names, rotation = 45, fontsize=4, ha = 'right', va = 'center', rotation_mode = 'anchor')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticklabels(class_names, rotation = 45, fontsize=4, ha = 'right', va = 'center', rotation_mode = 'anchor')
ax.xaxis.set_ticks_position('bottom')
ax.set_yticklabels(questions, rotation = 45, fontsize=4, ha = 'right', va = 'center', rotation_mode = 'anchor')

ax.set_xlabel('Class Names')
ax.set_ylabel('Questions')
ax.set_title('Correlation Matrix')

fig.tight_layout(pad=10.0)
plt.show()

generate_histograms = False

if generate_histograms:
    fig, ax = plt.subplots(1,1, figsize=(10,3), dpi = 300)
            
    n, bins, patches = ax.hist(class_labels, np.arange(9)-1.5, color='gold', ec = 'k')

    #Set the ticks to be at the centres of the bins.
    ax.set_xticks(bins+0.5)

    # #label bins and place frequency above them
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]

    for n, x in zip(n, bin_centers):
        # Label the frequency
        ax.annotate(str(int(n)), xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, 175), textcoords='offset points', va='top', ha='center', color = 'k')

    ax.set_title('Bar Chart of Galaxy Classes', pad =24)
    plt.xlabel('Galaxy Classes')
    plt.ylabel('Frequency (log scale)')

    # Add class names below x axis numbers
    class_names = ['Artifact', 'Smooth Inbetween', 'Smooth Round', 'Smooth Cigar', 'Edge on Disk', 'Unbarred Spiral', 'Barred Spiral', 'Featured without bar or spiral', None]

    # Set the fontsize for x tick labels
    ax.tick_params(axis='x', which='major', labelsize=4)
    ax.set_xticklabels(class_names)
    ax.set_xticks(ax.get_xticks()[ax.get_xticks() != 7])

    # Set the y-axis to log scale
    ax.set_yscale('log')

    plt.show()

    # Histogram of smooth or featured

    smooth_or_featured = df['smooth-or-featured-gz2_semantic']

    fig, ax = plt.subplots(1,1, figsize=(10,3), dpi = 300)

    n, bins, patches = ax.hist(smooth_or_featured, np.arange(5)-0.5, color='lightgray', ec = 'k')

    # Set the y-axis to log scale
    ax.set_yscale('log')

    # Set the ticks to be at the centres of the bins.
    ax.set_xticks(bins+0.5)

    # Label bins and place frequency above them
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]

    for n, x in zip(n, bin_centers):
        # Label the frequency
        ax.annotate(str(int(n)), xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, 175), textcoords='offset points', va='top', ha='center', color = 'k')
        
    ax.set_title('Is the galaxy smooth or featured?', pad =24)
    plt.xlabel('Answers')
    plt.ylabel('Frequency (log scale)')

    # Add class names below x axis numbers
    class_names = ['Smooth', 'Featured or Disk', 'Deadlock', 'Artifact', None]

    # Set the fontsize for x tick labels
    ax.set_xticklabels(class_names)
    ax.tick_params(axis='x', which='major', labelsize=4)
    ax.set_xticks(ax.get_xticks()[ax.get_xticks() != 4])

    plt.show()

    # Histogram of disk edge on

    disk_edge_on = df['disk-edge-on-gz2_semantic']

    class_names = ['-','No', 'Yes', 'Deadlock', None]

    fig, ax = plt.subplots(1,1, figsize=(10,3), dpi = 300)

    n, bins, patches = ax.hist(disk_edge_on, np.arange(5)-0.5, color='palegreen', ec = 'k')

    # Set the y-axis to log scale
    ax.set_yscale('log')

    # Set the ticks to be at the centres of the bins.
    ax.set_xticks(bins+0.5)

    # Label bins and place frequency above them
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]

    for n, x in zip(n, bin_centers):
        # Label the frequency
        ax.annotate(str(int(n)), xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, 175), textcoords='offset points', va='top', ha='center', color = 'k')

    ax.set_title('Can the galaxy be seen as edge on?', pad =24)
    ax.tick_params(axis='x', which='major', labelsize=4)
    ax.set_xticklabels(class_names)
    ax.set_xticks(ax.get_xticks()[ax.get_xticks() != 4])

    plt.xlabel('Answers')
    plt.ylabel('Frequency (log scale)')
    plt.show()

    # Histogram of has spiral arms

    has_spiral_arms = df['has-spiral-arms-gz2_semantic']

    class_names = ['-','No', 'Yes', 'Deadlock', None]

    fig, ax = plt.subplots(1,1, figsize=(10,3), dpi = 300)

    n, bins, patches = ax.hist(has_spiral_arms, np.arange(5)-0.5, color='skyblue', ec = 'k')

    # Set the y-axis to log scale
    ax.set_yscale('log')

    # Set the ticks to be at the centres of the bins.
    ax.set_xticks(bins+0.5)

    # Label bins and place frequency above them
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]

    for n, x in zip(n, bin_centers):
        # Label the frequency
        ax.annotate(str(int(n)), xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, 175), textcoords='offset points', va='top', ha='center', color = 'k')

    ax.set_title('Does the galaxy have spiral arms?', pad =24)
    ax.tick_params(axis='x', which='major', labelsize=4)
    ax.set_xticklabels(class_names)
    ax.set_xticks(ax.get_xticks()[ax.get_xticks() != 4])

    plt.xlabel('Answers')
    plt.ylabel('Frequency (log scale)')
    plt.show()

    # Histogram of bar

    bar = df['bar-gz2_semantic']

    class_names = ['-','No', 'Yes', 'Deadlock', None]

    fig, ax = plt.subplots(1,1, figsize=(10,3), dpi = 300)

    n, bins, patches = ax.hist(bar, np.arange(5)-0.5, color='skyblue', ec = 'k')

    # Set the y-axis to log scale
    ax.set_yscale('log')

    # Set the ticks to be at the centres of the bins.
    ax.set_xticks(bins+0.5)

    # Label bins and place frequency above them
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]

    for n, x in zip(n, bin_centers):
        # Label the frequency
        ax.annotate(str(int(n)), xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, 175), textcoords='offset points', va='top', ha='center', color = 'k')
        
    ax.set_title('Does the galaxy have a bar?', pad =24)
    ax.tick_params(axis='x', which='major', labelsize=4)
    ax.set_xticklabels(class_names)
    ax.set_xticks(ax.get_xticks()[ax.get_xticks() != 4])

    plt.xlabel('Answers')
    plt.ylabel('Frequency (log scale)')
    plt.show()

    # histogram of bulge size

    bulge_size = df['bulge-size-gz2_semantic']

    class_names = ['-','Dominant', 'Just Noticable', 'Obvious', 'Deadlock', None]

    fig, ax = plt.subplots(1,1, figsize=(10,3), dpi = 300)

    n, bins, patches = ax.hist(bulge_size, np.arange(6)-0.5, color='skyblue', ec = 'k')

    # Set the y-axis to log scale
    ax.set_yscale('log')

    # Set the ticks to be at the centres of the bins.
    ax.set_xticks(bins+0.5)

    # Label bins and place frequency above them
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]

    for n, x in zip(n, bin_centers):
        # Label the frequency
        ax.annotate(str(int(n)), xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, 175), textcoords='offset points', va='top', ha='center', color = 'k')
        
    ax.set_title("Does the Galaxy have a central bulge? If so, what it's size?", pad =24)
    ax.tick_params(axis='x', which='major', labelsize=4)
    ax.set_xticklabels(class_names)
    ax.set_xticks(ax.get_xticks()[ax.get_xticks() != 5])

    plt.xlabel('Answers')
    plt.ylabel('Frequency (log scale)')
    plt.show()

    # histogram of something odd

    something_odd = df['something-odd-gz2_semantic']

    class_names = ['No','Yes', 'Deadlock', None]

    fig, ax = plt.subplots(1,1, figsize=(10,3), dpi = 300)

    n, bins, patches = ax.hist(something_odd, np.arange(4)-0.5, color='palegreen', ec = 'k')

    # Set the y-axis to log scale
    ax.set_yscale('log')

    # Set the ticks to be at the centres of the bins.
    ax.set_xticks(bins+0.5)

    # Label bins and place frequency above them
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]

    for n, x in zip(n, bin_centers):
        # Label the frequency
        ax.annotate(str(int(n)), xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, 175), textcoords='offset points', va='top', ha='center', color = 'k')

    ax.set_title('Does the image of the galaxy have something odd in it?', pad =24)
    ax.tick_params(axis='x', which='major', labelsize=4)
    ax.set_xticklabels(class_names)
    ax.set_xticks(ax.get_xticks()[ax.get_xticks() != 3])

    plt.xlabel('Answers')
    plt.ylabel('Frequency (log scale)')
    plt.show()

    # histogram of how rounded

    how_rounded = df['how-rounded-gz2_semantic']

    class_names = ['Round','In-between', '-', 'Cigar', 'Deadlock',None]

    fig, ax = plt.subplots(1,1, figsize=(10,3), dpi = 300)

    n, bins, patches = ax.hist(how_rounded, np.arange(6)-0.5, color='palegreen', ec = 'k')

    # Set the y-axis to log scale
    ax.set_yscale('log')

    # Set the ticks to be at the centres of the bins.
    ax.set_xticks(bins+0.5)

    # Label bins and place frequency above them
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]

    for n, x in zip(n, bin_centers):
        # Label the frequency
        ax.annotate(str(int(n)), xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, 175), textcoords='offset points', va='top', ha='center', color = 'k')

    ax.set_title('How rounded is the galaxy?', pad =24)
    ax.tick_params(axis='x', which='major', labelsize=4)
    ax.set_xticklabels(class_names)
    ax.set_xticks(ax.get_xticks()[ax.get_xticks() != 5])

    plt.xlabel('Answers')
    plt.ylabel('Frequency (log scale)')
    plt.show()

    # histogram of bulge shape

    bulge_shape = df['bulge-shape-gz2_semantic']

    class_names = ['-', 'Round', 'Deadlock', 'None', 'Boxy', None]

    fig, ax = plt.subplots(1,1, figsize=(10,3), dpi = 300)

    n, bins, patches = ax.hist(bulge_shape, np.arange(6)-0.5, color='skyblue', ec = 'k')

    # Set the y-axis to log scale

    ax.set_yscale('log')

    # Set the ticks to be at the centres of the bins.
    ax.set_xticks(bins+0.5)

    # Label bins and place frequency above them
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]

    for n, x in zip(n, bin_centers):
        # Label the frequency
        ax.annotate(str(int(n)), xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, 175), textcoords='offset points', va='top', ha='center', color = 'k')

    ax.set_title('What is the shape of the bulge in the galaxy?', pad =24)
    ax.set_xticklabels(class_names)
    ax.tick_params(axis='x', which='major', labelsize=4)
    ax.set_xticks(ax.get_xticks()[ax.get_xticks() != 5])

    plt.xlabel('Answers')
    plt.ylabel('Frequency (log scale)')
    plt.show()

    # histogram of spiral winding

    spiral_winding = df['spiral-winding-gz2_semantic']

    class_names = ['-', 'Deadlock', 'Loose', 'Medium', 'Tight', None]

    fig, ax = plt.subplots(1,1, figsize=(10,3), dpi = 300)

    n, bins, patches = ax.hist(spiral_winding, np.arange(6)-0.5, color='plum', ec = 'k')

    # Set the y-axis to log scale
    ax.set_yscale('log')

    # Set the ticks to be at the centres of the bins.
    ax.set_xticks(bins+0.5)

    # Label bins and place frequency above them
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]

    for n, x in zip(n, bin_centers):
        # Label the frequency
        ax.annotate(str(int(n)), xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, 175), textcoords='offset points', va='top', ha='center', color = 'k')
        
    ax.set_title('How tightly wound are the spiral arms?', pad =24)
    ax.set_xticklabels(class_names, fontsize=4)
    ax.set_xticks(ax.get_xticks()[ax.get_xticks() != 5])

    plt.xlabel('Answers')
    plt.ylabel('Frequency (log scale)')
    plt.show()

    # histogram of spiral arm count

    spiral_arm_count = df['spiral-arm-count-gz2_semantic']

    class_names = ['-', '1', '2', '4', "Can't tell", '3', 'Deadlock', 'More than 4', None]

    fig, ax = plt.subplots(1,1, figsize=(10,3), dpi = 300)

    n, bins, patches = ax.hist(spiral_arm_count, np.arange(9)-0.5, color='plum', ec = 'k')

    # Set the y-axis to log scale
    ax.set_yscale('log')

    # Set the ticks to be at the centres of the bins.
    ax.set_xticks(bins+0.5)

    # Label bins and place frequency above them
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]

    for n, x in zip(n, bin_centers):
        # Label the frequency
        ax.annotate(str(int(n)), xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, 175), textcoords='offset points', va='top', ha='center', color = 'k')
        
    ax.set_title('How many spiral arms are there?', pad =24)
    ax.tick_params(axis='x', which='major', labelsize=4)
    ax.set_xticklabels(class_names)
    ax.set_xticks(ax.get_xticks()[ax.get_xticks() != 8])

    plt.xlabel('Answers')
    plt.ylabel('Frequency (log scale)')
    plt.show()

    #Plot answers to every classifying question in the decision tree (index smooth-or-featured-gz2_smooth to spiral-arm-count-gz2_cant_tell) as a histogram

    smooth = df['smooth-or-featured-gz2_smooth']
    featured = df['smooth-or-featured-gz2_featured-or-disk']
    artifact = df['smooth-or-featured-gz2_artifact']
    edge_on_yes = df['disk-edge-on-gz2_yes']
    edge_on_no = df['disk-edge-on-gz2_no']
    bar_yes = df['bar-gz2_yes']
    bar_no = df['bar-gz2_no']
    has_spiral_arms_yes = df['has-spiral-arms-gz2_yes']
    has_spiral_arms_no = df['has-spiral-arms-gz2_no']
    bulge_no = df['bulge-size-gz2_no']
    bulge_just_noticable = df['bulge-size-gz2_just-noticeable']
    bulge_obvious = df['bulge-size-gz2_obvious']
    bulge_dominant = df['bulge-size-gz2_dominant']
    something_odd_yes = df['something-odd-gz2_yes']
    something_odd_no = df['something-odd-gz2_no']
    round = df['how-rounded-gz2_round']
    in_between = df['how-rounded-gz2_in-between']
    cigar = df['how-rounded-gz2_cigar']
    bulge_round = df['bulge-shape-gz2_round']
    bulge_boxy = df['bulge-shape-gz2_boxy']
    bulge_no_bulge = df['bulge-shape-gz2_no-bulge']
    spiral_winding_tight = df['spiral-winding-gz2_tight']
    spiral_winding_medium = df['spiral-winding-gz2_medium']
    spiral_winding_loose = df['spiral-winding-gz2_loose']
    spiral_count_1 = df['spiral-arm-count-gz2_1']
    spiral_count_2 = df['spiral-arm-count-gz2_2']
    spiral_count_3 = df['spiral-arm-count-gz2_3']
    spiral_count_4 = df['spiral-arm-count-gz2_4']
    spiral_count_more_than_4 = df['spiral-arm-count-gz2_more-than-4']
    spiral_count_cant_tell = df['spiral-arm-count-gz2_cant-tell']

    data = [smooth, featured, artifact, edge_on_yes, edge_on_no, bar_yes, bar_no, has_spiral_arms_yes, has_spiral_arms_no, bulge_no, bulge_just_noticable, 
            bulge_obvious, bulge_dominant, something_odd_yes, something_odd_no, round, in_between, cigar, bulge_round, bulge_boxy, bulge_no_bulge, spiral_winding_tight, 
            spiral_winding_medium, spiral_winding_loose, spiral_count_1, spiral_count_2, spiral_count_3, spiral_count_4, spiral_count_more_than_4, spiral_count_cant_tell]

    class_names = ['Smooth', 'Featured', 'Artifact', 'Edge on Disk', 'Not Edge on Disk', 'Bar', 'Not Bar', 'Has Spiral Arms', 'No Spiral Arms', 'No Bulge', 'Just Noticable Bulge',
                    'Obvious Bulge', 'Dominant Bulge', 'Something Odd', 'Nothing Odd', 'Round', 'In between', 'Cigar', 'Round Bulge', 'Boxy Bulge', 'No Bulge', 'Tight Winding', 
                    'Medium Winding', 'Loose Winding', '1 Arm', '2 Arms', '3 Arms', '4  Arms', 'More than 4 Arms', "Can't Tell no. Arms"]

    fig, axs = plt.subplots(len(data), 1, figsize=(10, 20), sharex=True)

    cmap = plt.cm.hsv

    for i, ax in enumerate(axs):
        if i == 0:
            ax.set_title('Distribution of vote counts for each question (relative frequency)', fontsize=10, color='k', ha='center', va='center', fontweight='bold')
            
        ax.hist(data[i], bins=100, color=cmap(i/len(data)), edgecolor='black')  # Add edgecolor='black'
        ax.set_ylabel(f'{class_names[i]}', rotation=0, labelpad=20, fontsize=8, fontweight='bold', ha='right')
        
        # Remove spines and ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.tick_params(axis='y', which='both', direction='out')
        ax.set_xlim(0, 70)
        ax.set_yticks([])  # Add this line to remove y ticks

    plt.xlabel('Number of votes')

    plt.tight_layout()  # Add this line to remove gaps between subplots
    plt.show()
    
    smooth_frac = df['smooth-or-featured-gz2_smooth_fraction']
    disk_frac = df['smooth-or-featured-gz2_featured-or-disk_fraction']
    artifact_frac = df['smooth-or-featured-gz2_artifact_fraction']
    disk_edge_on_yes_frac = df['disk-edge-on-gz2_yes_fraction']
    disk_edge_on_no_frac = df['disk-edge-on-gz2_no_fraction']
    has_spiral_arms_yes_frac = df['has-spiral-arms-gz2_yes_fraction']
    has_spiral_arms_no_frac = df['has-spiral-arms-gz2_no_fraction']
    bar_yes_frac = df['bar-gz2_yes_fraction']
    bar_no_frac = df['bar-gz2_no_fraction']
    bulge_size_no_frac = df['bulge-size-gz2_no_fraction']
    bulge_size_just_noticable_frac = df['bulge-size-gz2_just-noticeable_fraction']
    bulge_size_obvious_frac = df['bulge-size-gz2_obvious_fraction']
    bulge_size_dominant_frac = df['bulge-size-gz2_dominant_fraction']
    something_odd_yes_frac = df['something-odd-gz2_yes_fraction']
    something_odd_no_frac = df['something-odd-gz2_no_fraction']
    how_rounded_round_frac = df['how-rounded-gz2_round_fraction']
    how_rounded_in_between_frac = df['how-rounded-gz2_in-between_fraction']
    how_rounded_cigar_frac = df['how-rounded-gz2_cigar_fraction']
    bulge_shape_round_frac = df['bulge-shape-gz2_round_fraction']
    bulge_shape_boxy_frac = df['bulge-shape-gz2_boxy_fraction']
    bulge_shape_no_bulge_frac = df['bulge-shape-gz2_no-bulge_fraction']
    spiral_winding_tight_frac = df['spiral-winding-gz2_tight_fraction']
    spiral_winding_medium_frac = df['spiral-winding-gz2_medium_fraction']
    spiral_winding_loose_frac = df['spiral-winding-gz2_loose_fraction']
    spiral_arm_count_1_frac = df['spiral-arm-count-gz2_1_fraction']
    spiral_arm_count_2_frac = df['spiral-arm-count-gz2_2_fraction']
    spiral_arm_count_3_frac = df['spiral-arm-count-gz2_3_fraction']
    spiral_arm_count_4_frac = df['spiral-arm-count-gz2_4_fraction']
    spiral_arm_count_more_than_4_frac = df['spiral-arm-count-gz2_more-than-4_fraction']
    spiral_arm_count_cant_tell_frac = df['spiral-arm-count-gz2_cant-tell_fraction']

    data = [smooth_frac, disk_frac, artifact_frac, disk_edge_on_yes_frac, disk_edge_on_no_frac, has_spiral_arms_yes_frac, has_spiral_arms_no_frac, bar_yes_frac, bar_no_frac,
            bulge_size_no_frac, bulge_size_just_noticable_frac, bulge_size_obvious_frac, bulge_size_dominant_frac, something_odd_yes_frac, something_odd_no_frac, how_rounded_round_frac,
            how_rounded_in_between_frac, how_rounded_cigar_frac, bulge_shape_round_frac, bulge_shape_boxy_frac, bulge_shape_no_bulge_frac, spiral_winding_tight_frac, spiral_winding_medium_frac,
            spiral_winding_loose_frac, spiral_arm_count_1_frac, spiral_arm_count_2_frac, spiral_arm_count_3_frac, spiral_arm_count_4_frac, spiral_arm_count_more_than_4_frac, spiral_arm_count_cant_tell_frac]

    class_names = ['Smooth', 'Featured', 'Artifact', 'Edge on Disk', 'Not Edge on Disk', 'Bar', 'Not Bar', 'Has Spiral Arms', 'No Spiral Arms', 'No Bulge', 'Just Noticable Bulge',
                    'Obvious Bulge', 'Dominant Bulge', 'Something Odd', 'Nothing Odd', 'Round', 'In between', 'Cigar', 'Round Bulge', 'Boxy Bulge', 'No Bulge', 'Tight Winding', 
                    'Medium Winding', 'Loose Winding', '1 Arm', '2 Arms', '3 Arms', '4  Arms', 'More than 4 Arms', "Can't Tell no. Arms"]

    fig, axs = plt.subplots(len(data), 1, figsize=(10, 20), sharex=True)

    cmap = plt.cm.hsv

    for i, ax in enumerate(axs):
        if i == 0:
            ax.set_title('Confidence in answers for each question (relative frequency)', fontsize=10, color='k', ha='center', va='center', fontweight='bold')
            
        ax.hist(data[i], bins=100, color=cmap(i/len(data)), edgecolor='black')  # Add edgecolor='black'
        ax.set_ylabel(f'{class_names[i]}', rotation=0, labelpad=20, fontsize=8, fontweight='bold', ha='right')

        # Remove spines and ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.tick_params(axis='y', which='both', direction='out')
        ax.set_yticks([])  # Add this line to remove y ticks
        ax.set_xlim(0, 1)

    plt.xlabel('Probability')

    plt.tight_layout()  # Add this line to remove gaps between subplots
    plt.show()
    
    smooth_or_featured_total = df['smooth-or-featured-gz2_total-votes']
    disk_edge_on_total = df['disk-edge-on-gz2_total-votes']
    spiral_arms_total = df['has-spiral-arms-gz2_total-votes']
    bar_total = df['bar-gz2_total-votes']
    bulge_size_total = df['bulge-size-gz2_total-votes']
    something_odd_total = df['something-odd-gz2_total-votes']
    how_rounded_total = df['how-rounded-gz2_total-votes']
    bulge_shape_total = df['bulge-shape-gz2_total-votes']
    spiral_winding_total = df['spiral-winding-gz2_total-votes']
    spiral_arm_count_total = df['spiral-arm-count-gz2_total-votes']

    data = [smooth_or_featured_total, disk_edge_on_total, spiral_arms_total, bar_total, bulge_size_total, something_odd_total, how_rounded_total, bulge_shape_total, spiral_winding_total, spiral_arm_count_total]

    class_names = ['Smooth or Featured', 'Disk Edge On', 'Has Spiral Arms', 'Bar', 'Bulge Size', 'Something Odd', 'How Rounded', 'Bulge Shape', 'Spiral Winding', 'Spiral Arm Count']

    fig, axs = plt.subplots(len(data), 1, figsize=(10, 20), sharex=True)

    cmap = plt.cm.hsv

    for i, ax in enumerate(axs):
        if i == 0:
            ax.set_title('Toal Votes for each question', fontsize=10, color='k', ha='center', va='center', fontweight='bold')
            
        ax.hist(data[i], bins=100, color=cmap(i/len(data)), edgecolor='black')  # Add edgecolor='black'
        ax.set_ylabel(f'{class_names[i]}', rotation=0, labelpad=20, fontsize=8, fontweight='bold', ha='right')

        # Remove spines and ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.tick_params(axis='y', which='both', direction='out')
        ax.set_xlim(0, 70)

    plt.xlabel('Number of votes')

    plt.tight_layout()  # Add this line to remove gaps between subplots
    plt.show()

    mean_confidence_smooth_or_featured_smooth = df['smooth-or-featured-gz2_smooth_fraction'].mean()
    mean_confidence_smooth_or_featured_featured = df['smooth-or-featured-gz2_featured-or-disk_fraction'].mean()
    mean_confidence_smooth_or_featured_artifact = df['smooth-or-featured-gz2_artifact_fraction'].mean()
    mean_confidence_disk_edge_on_yes = df['disk-edge-on-gz2_yes_fraction'].mean()
    mean_confidence_disk_edge_on_no = df['disk-edge-on-gz2_no_fraction'].mean()
    mean_confidence_bar_yes = df['bar-gz2_yes_fraction'].mean()
    mean_confidence_bar_no = df['bar-gz2_no_fraction'].mean()
    mean_confidence_has_spiral_arms_yes = df['has-spiral-arms-gz2_yes_fraction'].mean()
    mean_confidence_has_spiral_arms_no = df['has-spiral-arms-gz2_no_fraction'].mean()
    mean_confidence_bulge_size_no = df['bulge-size-gz2_no_fraction'].mean()
    mean_confidence_bulge_size_just_noticable = df['bulge-size-gz2_just-noticeable_fraction'].mean()
    mean_confidence_bulge_size_obvious = df['bulge-size-gz2_obvious_fraction'].mean()
    mean_confidence_bulge_size_dominant = df['bulge-size-gz2_dominant_fraction'].mean()
    mean_confidence_something_odd_yes = df['something-odd-gz2_yes_fraction'].mean()
    mean_confidence_something_odd_no = df['something-odd-gz2_no_fraction'].mean()
    mean_confidence_how_rounded_round = df['how-rounded-gz2_round_fraction'].mean()
    mean_confidence_how_rounded_in_between = df['how-rounded-gz2_in-between_fraction'].mean()
    mean_confidence_how_rounded_cigar = df['how-rounded-gz2_cigar_fraction'].mean()
    mean_confidence_bulge_shape_round = df['bulge-shape-gz2_round_fraction'].mean()
    mean_confidence_bulge_shape_boxy = df['bulge-shape-gz2_boxy_fraction'].mean()
    mean_confidence_bulge_shape_no_bulge = df['bulge-shape-gz2_no-bulge_fraction'].mean()
    mean_confidence_spiral_winding_tight = df['spiral-winding-gz2_tight_fraction'].mean()
    mean_confidence_spiral_winding_medium = df['spiral-winding-gz2_medium_fraction'].mean()
    mean_confidence_spiral_winding_loose = df['spiral-winding-gz2_loose_fraction'].mean()
    mean_confidence_spiral_arm_count_1 = df['spiral-arm-count-gz2_1_fraction'].mean()
    mean_confidence_spiral_arm_count_2 = df['spiral-arm-count-gz2_2_fraction'].mean()
    mean_confidence_spiral_arm_count_3 = df['spiral-arm-count-gz2_3_fraction'].mean()
    mean_confidence_spiral_arm_count_4 = df['spiral-arm-count-gz2_4_fraction'].mean()
    mean_confidence_spiral_arm_count_more_than_4 = df['spiral-arm-count-gz2_more-than-4_fraction'].mean()
    mean_confidence_spiral_arm_count_cant_tell = df['spiral-arm-count-gz2_cant-tell_fraction'].mean()

    data = [mean_confidence_smooth_or_featured_smooth, mean_confidence_smooth_or_featured_featured, mean_confidence_smooth_or_featured_artifact,
            mean_confidence_disk_edge_on_yes, mean_confidence_disk_edge_on_no, mean_confidence_bar_yes, mean_confidence_bar_no, mean_confidence_has_spiral_arms_yes, 
            mean_confidence_has_spiral_arms_no, mean_confidence_bulge_size_no, mean_confidence_bulge_size_just_noticable, mean_confidence_bulge_size_obvious, 
            mean_confidence_bulge_size_dominant, mean_confidence_something_odd_yes, mean_confidence_something_odd_no, mean_confidence_how_rounded_round, 
            mean_confidence_how_rounded_in_between, mean_confidence_how_rounded_cigar, mean_confidence_bulge_shape_round, mean_confidence_bulge_shape_boxy, 
            mean_confidence_bulge_shape_no_bulge, mean_confidence_spiral_winding_tight, mean_confidence_spiral_winding_medium, mean_confidence_spiral_winding_loose, 
            mean_confidence_spiral_arm_count_1, mean_confidence_spiral_arm_count_2, mean_confidence_spiral_arm_count_3, mean_confidence_spiral_arm_count_4, 
            mean_confidence_spiral_arm_count_more_than_4, mean_confidence_spiral_arm_count_cant_tell]

    class_names = ['Smooth', 'Featured or Disk', 'Artifact', 'Disk Edge On', 'Disk Not Edge On', 'Bar', 'No Bar', 'Has Spiral Arms', 'No Spiral Arms', 'No Bulge', 
                'Just Noticable Bulge', 'Obvious Bulge', 'Dominant Bulge', 'Something Odd', 'Nothing odd', 'Round', 'In Between', 
                'Cigar', 'Bulge Shape Round', 'Bulge Shape Boxy', 'Bulge Shape None', 'Spiral Winding Tight', 'Spiral Winding Medium', 'Spiral Winding Loose', 
                'Spiral Arm no. 1', 'Spiral Arm no. 2', 'Spiral Arm no. 3', 'Spiral Arm no. 4', 'Spiral Arm no. > 4', 'Spiral Arm no. ???']

    #plot the mean confidence of answers as a bar chart

    fig, ax = plt.subplots(1,1, figsize=(10,6), dpi = 300)

    bars = ax.bar(class_names, data, color='tomato', edgecolor='black')

    ax.set_ylabel('Mean Confidence')
    ax.set_xlabel('Question')
    ax.set_title('Mean Confidence of Answers')
    plt.xticks(rotation=(45), ha='right', fontsize=4)
    plt.tight_layout()
    plt.show()

    filtered_data = [d for d in data if d >= 0.5]
    filtered_class_names = [cn for d, cn in zip(data, class_names) if d > 0.5]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)

    bars = ax.bar(filtered_class_names, filtered_data, color='aquamarine', edgecolor='black')

    ax.set_ylabel('Mean Confidence')
    ax.set_xlabel('Question')
    ax.set_title('Questions answered with a mean confidence of 0.5 or higher')
    plt.xticks(rotation=(45), ha='right', fontsize=4)
    plt.tight_layout()
    plt.show()

    filtered_data = [d for d in data if d <= 0.2]
    filtered_class_names = [cn for d, cn in zip(data, class_names) if d <= 0.2]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)

    bars = ax.bar(filtered_class_names, filtered_data, color='lightgoldenrodyellow', edgecolor='black')

    ax.set_ylabel('Mean Confidence')
    ax.set_xlabel('Question')
    ax.set_title('Questions answered with a mean confidence of 0.2 or less')
    plt.xticks(rotation=(45), ha='right', fontsize=4)
    plt.tight_layout()
    plt.show()