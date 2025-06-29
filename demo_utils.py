import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from matplotlib.widgets import Button
from plots import format_value
import cv2
import colors

def get_resnet_imagenet_preprocess(resize_to=(448, 448)):
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(resize_to),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess

color1 = [1.0, 0.0, 0.0, 0.5]
color2 = [0.0, 0.0, 1.0, 0.5]

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0.5
    for idx, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        if idx==0:
            color_mask = color2
        else:
            color_mask = color1
        img[m] = color_mask
    ax.imshow(img)


def bar(contributions, feature_names,  max_display=10, show=True, title=None, fontsize=13):

    values = contributions

    # build our auto xlabel based on the transform history of the Explanation object
    
    xlabel = "Concept activation"

    # ensure we at least have default feature names
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(values[0]))])

    # determine how many top features we will plot
    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(values))
    max_display = min(max_display, num_features)

    orig_inds = [i for i in range(len(values))]
    
    # feature_order = np.argsort(np.abs(values), 0)[::-1]
    feature_order = np.argsort(values, 0)[::-1]
    # here we build our feature names, accounting for the fact that some features might be merged together
    feature_inds = feature_order[:max_display]
    y_pos = np.arange(len(feature_inds), 0, -1)
    feature_names_new = []
    for pos,inds in enumerate(orig_inds):
        feature_names_new.append(feature_names[inds])
    feature_names = feature_names_new
    
    # build our y-tick labels
    yticklabels = []
    for i in feature_inds:
        yticklabels.append(feature_names[i])

    # compute our figure size based on how many features we are showing
    row_height = 0.55
    plt.gcf().set_size_inches(8, num_features * row_height  + 1.5)#* np.sqrt(len(values))

    # if negative values are present then we draw a vertical line to mark 0, otherwise the axis does this for us...
    negative_values_present = np.sum(values[feature_order[:num_features]] < 0) > 0
    if negative_values_present:
        plt.axvline(0, 0, 1, color="#000000", linestyle="-", linewidth=1, zorder=1)

    # draw the bars
    patterns = (None, '\\\\', '++', 'xx', '////', '*', 'o', 'O', '.', '-')
    total_width = 0.7
    bar_width = total_width# / len(values)
    
    #ypos_offset = - ((i - len(values) / 2) * bar_width + bar_width / 2)
    plt.barh(
        y_pos, values[feature_inds],
        bar_width, align='center',
        color=[colors.blue_rgb if values[feature_inds[j]] <= 0 else colors.red_rgb for j in range(len(y_pos))],
        hatch=patterns[0], edgecolor=(1,1,1,0.8), label=None#f"{cohort_labels[i]} [{cohort_sizes[i] if i < len(cohort_sizes) else None}]"
    )

    # draw the yticks (the 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks)
    plt.yticks(list(y_pos) + list(y_pos + 1e-8), yticklabels + [l.split('=')[-1] for l in yticklabels], fontsize=fontsize)

    xlen = plt.xlim()[1] - plt.xlim()[0]
    fig = plt.gcf()
    ax = plt.gca()
    #xticks = ax.get_xticks()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    bbox_to_xscale = xlen/width

    
    for j in range(len(y_pos)):
        ind = feature_order[j]
        if values[ind] < 0:
            plt.text(
                values[ind] - (5/72)*bbox_to_xscale, y_pos[j], format_value(values[ind], '%+0.02f'),
                horizontalalignment='right', verticalalignment='center', color=colors.blue_rgb,
                fontsize=fontsize
            )
        else:
            plt.text(
                values[ind] + (5/72)*bbox_to_xscale, y_pos[j], format_value(values[ind], '%+0.02f'),
                horizontalalignment='left', verticalalignment='center', color=colors.red_rgb,
                fontsize=fontsize
            )

    # put horizontal lines for each feature row
    for i in range(num_features):
        plt.axhline(i+1, color="#888888", lw=0.5, dashes=(1, 5), zorder=-1)
    
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    if negative_values_present:
        plt.gca().spines['left'].set_visible(False)
    plt.gca().tick_params('x', labelsize=fontsize)

    xmin,xmax = plt.gca().get_xlim()
    
    if negative_values_present:
        plt.gca().set_xlim(xmin - (xmax-xmin)*0.1, xmax + (xmax-xmin)*0.1)
    else:
        plt.gca().set_xlim(xmin, xmax + (xmax-xmin)*0.1)
    
    plt.xlabel(xlabel, fontsize=fontsize)
    if title:
        plt.title(title, fontsize=fontsize)

    plt.tight_layout()
    if show:
        plt.show()

def bar2(contributions, feature_names, max_display=10, show=True, title=None, fontsize=13, bar_colors_rgba=None, label_colors_rgba=None):
    values = contributions

    # build our auto xlabel based on the transform history of the Explanation object
    xlabel = "Concept activations"

    # ensure we at least have default feature names
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(values[0]))])

    # determine how many top features we will plot
    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(values))
    max_display = min(max_display, num_features)

    orig_inds = [i for i in range(len(values))]
    feature_order = np.argsort(values, 0)[::-1]
    feature_inds = feature_order[:max_display]
    y_pos = np.arange(len(feature_inds), 0, -1)
    
    # New feature names array
    feature_names_new = []
    for pos, inds in enumerate(orig_inds):
        feature_names_new.append(feature_names[inds])
    feature_names = feature_names_new

    # y-tick labels
    yticklabels = [feature_names[i] for i in feature_inds]

    row_height = 0.55
    plt.gcf().set_size_inches(8, num_features * row_height + 1.5)

    negative_values_present = np.sum(values[feature_order[:num_features]] < 0) > 0
    if negative_values_present:
        plt.axvline(0, 0, 1, color="#000000", linestyle="-", linewidth=1, zorder=1)

    total_width = 0.7
    bar_width = total_width
    
    # Draw the bars with specified RGBA colors
    plt.barh(
        y_pos, values[feature_inds], bar_width, align='center',
        color=bar_colors_rgba if bar_colors_rgba else ['blue' if v <= 0 else 'red' for v in values[feature_inds]],
        edgecolor=(1, 1, 1, 0.8), label=None
    )

    plt.yticks(list(y_pos) + list(y_pos + 1e-8), yticklabels + [l.split('=')[-1] for l in yticklabels], fontsize=fontsize)

    xlen = plt.xlim()[1] - plt.xlim()[0]
    fig = plt.gcf()
    ax = plt.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    bbox_to_xscale = xlen / width

    # Set the labels with RGBA colors
    for j in range(len(y_pos)):
        ind = feature_order[j]
        if values[ind] < 0:
            plt.text(
                values[ind] - (5/72)*bbox_to_xscale, y_pos[j], format_value(values[ind], '%+0.02f'),
                horizontalalignment='right', verticalalignment='center',
                color=label_colors_rgba if label_colors_rgba else 'blue',
                fontsize=fontsize
            )
        else:
            plt.text(
                values[ind] + (5/72)*bbox_to_xscale, y_pos[j], format_value(values[ind], '%+0.02f'),
                horizontalalignment='left', verticalalignment='center',
                color=label_colors_rgba if label_colors_rgba else 'red',
                fontsize=fontsize
            )

    # Optional horizontal lines
    for i in range(num_features):
        plt.axhline(i + 1, color="#888888", lw=0.5, dashes=(1, 5), zorder=-1)

    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    if negative_values_present:
        plt.gca().spines['left'].set_visible(False)
    plt.gca().tick_params('x', labelsize=fontsize)

    xmin, xmax = plt.gca().get_xlim()
    if negative_values_present:
        plt.gca().set_xlim(xmin - (xmax - xmin) * 0.1, xmax + (xmax - xmin) * 0.1)
    else:
        plt.gca().set_xlim(xmin, xmax + (xmax - xmin) * 0.1)

    plt.xlabel(xlabel, fontsize=fontsize)
    if title:
        plt.title(title, fontsize=fontsize)

    plt.tight_layout()
    if show:
        plt.show()

    
class RegionSelector:
    def __init__(self, image_path):
        self.image = plt.imread(image_path)
        self.height, self.width = self.image.shape[:2]
        
        # Create a new figure with enough size
        plt.close('all')  # Close any existing figures
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        
        self.regions = []
        self.current_vertices = []
        self.current_path = None
        self.region_colors = []
        self.binary_masks = []
        self.is_finished = False
        
        # Define a list of distinct colors (RGBA format with alpha=0.5)
        self.distinct_colors = [
            (1.0, 0.0, 0.0, 0.5),    # Red
            (0.0, 0.8, 0.0, 0.5),    # Green
            (0.0, 0.0, 1.0, 0.5),    # Blue
            (1.0, 1.0, 0.0, 0.5),    # Yellow
            (1.0, 0.0, 1.0, 0.5),    # Magenta
            (0.0, 1.0, 1.0, 0.5),    # Cyan
            (1.0, 0.5, 0.0, 0.5),    # Orange
            (0.5, 0.0, 1.0, 0.5),    # Purple
            (0.0, 0.5, 0.5, 0.5),    # Teal
            (0.5, 0.5, 0.0, 0.5),    # Olive
            (0.8, 0.4, 0.2, 0.5),    # Brown
            (0.2, 0.7, 0.5, 0.5),    # Sea Green
            (1.0, 0.6, 0.8, 0.5),    # Pink
            (0.6, 0.2, 0.0, 0.5),    # Dark Orange
            (0.0, 0.4, 0.8, 0.5),    # Azure
        ]
        
        # Display image
        self.image_display = self.ax.imshow(self.image)
        self.ax.set_title("Click to add points, press Enter to complete region")
        
        # Create a button to finish all selections
        self.ax_finish = plt.axes([0.8, 0.01, 0.1, 0.05])
        self.btn_finish = Button(self.ax_finish, 'Finish')
        self.btn_finish.on_clicked(self.on_finish)
        
        # Event connections
        self.click_cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.key_cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        plt.tight_layout()
        
    def on_click(self, event):
        # Ignore clicks outside the image
        if event.inaxes != self.ax or self.is_finished:
            return
            
        # Add the clicked point to current vertices
        self.current_vertices.append((event.xdata, event.ydata))
        
        # Update the display to show the current path
        if len(self.current_vertices) > 1:
            if self.current_path:
                self.current_path.remove()
            self.current_path = self.ax.plot(
                [x for x, y in self.current_vertices],
                [y for x, y in self.current_vertices],
                '-o', color='white', alpha=0.7)[0]
            self.fig.canvas.draw_idle()
            print(f"Point added at ({event.xdata:.2f}, {event.ydata:.2f})")
    
    def on_key(self, event):
        if event.key == 'enter' and len(self.current_vertices) > 2 and not self.is_finished:
            # Get color for this region
            color_idx = len(self.regions) % len(self.distinct_colors)
            color = self.distinct_colors[color_idx]
            self.region_colors.append(color)
            
            # Create a polygon from the vertices
            poly = plt.Polygon(self.current_vertices, facecolor=color)
            self.ax.add_patch(poly)
            
            # Store the region
            self.regions.append(self.current_vertices.copy())  # Make a copy
            
            # Create binary mask for this region
            self.create_binary_mask(self.current_vertices)
            
            # Reset for next selection
            if self.current_path:
                self.current_path.remove()
                self.current_path = None
            self.current_vertices = []
            
            # Update the display
            self.fig.canvas.draw_idle()
            print(f"Region {len(self.regions)} completed with color {self.get_color_name(color_idx)}. Start a new region or click 'Finish'.")
    
    def get_color_name(self, color_idx):
        """Return a descriptive name for the color used"""
        color_names = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan", "Orange", 
                       "Purple", "Teal", "Olive", "Brown", "Sea Green", "Pink", 
                       "Dark Orange", "Azure"]
        return color_names[color_idx % len(color_names)]
    
    def create_binary_mask(self, vertices):
        # Create empty mask
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Convert vertices to integer numpy array
        vertices_array = np.array(vertices).round().astype(np.int32)
        
        # Fill the polygon
        cv2.fillPoly(mask, [vertices_array], 1)
        
        # Store the binary mask
        self.binary_masks.append(mask)
        
        print(f"Region {len(self.binary_masks)} created - mask shape: {mask.shape}")
    
    def on_finish(self, event):
        self.is_finished = True
        self.fig.canvas.mpl_disconnect(self.click_cid)
        self.fig.canvas.mpl_disconnect(self.key_cid)
        print("Selection finished. Displaying results...")
        self.display_results()
    
    def display_results(self):
        # Display original image and all binary masks
        n_masks = len(self.binary_masks)
        if n_masks == 0:
            print("No regions were selected.")
            return
            
        fig, axes = plt.subplots(1, n_masks + 1, figsize=(4 * (n_masks + 1), 4))
        
        # If only one mask, axes will be a single axis
        if n_masks == 0:
            axes = [axes]
        
        # Show original image with all regions
        axes[0].imshow(self.image)
        for i, vertices in enumerate(self.regions):
            color_idx = i % len(self.distinct_colors)
            poly = plt.Polygon(vertices, facecolor=self.distinct_colors[color_idx])
            axes[0].add_patch(poly)
        axes[0].set_title("All Regions")
        axes[0].axis('off')
        
        # Show individual binary masks
        for i, mask in enumerate(self.binary_masks):
            axes[i+1].imshow(mask, cmap='gray')
            axes[i+1].set_title(f"Region {i+1} - {self.get_color_name(i)}")
            axes[i+1].axis('off')
        
        plt.tight_layout()
    
    def get_masks(self):
        """Return all binary masks"""
        return self.binary_masks