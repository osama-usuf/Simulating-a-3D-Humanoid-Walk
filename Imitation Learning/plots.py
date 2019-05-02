import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#Averaged over 10 runs

#Expert Policy
e_returns = [10681.977439808608, 10684.058806093002, 10662.046016328366, 10630.068107251342, 10711.78551448152,
             10685.222856832534, 10720.310180859393, 10670.263209758907, 10672.30618495687, 10751.445999804253, 
             10669.486179009813, 10691.27757164101, 10683.975483083837, 10702.98855677533, 10725.537928790623, 
             10647.03211067541, 10728.600380752481, 10711.139887459247, 10716.559488611483, 10713.90938754227, 
             10751.18616769406, 10692.565754978927, 10677.258871448324, 10680.279248777793, 10630.995925367442]
e_mean = 10691.691090351314
e_std = 31.45473030855253

#Regular Policy
r_returns = [10500.410892326676, 4270.5607988287975, 9399.584489380371, 839.8194546027222, 10527.81698393371, 
            391.1945642666502, 1419.2515775028169, 10569.846513667224, 10599.69840059753, 6318.383947339268, 
            3453.025106526041, 4947.753171279084, 2942.5777591360447, 10474.104790572199, 10432.810490616648, 
            10645.350262278296, 9115.539483890085, 1417.6515567863632, 10575.180247376404, 10486.377120439352, 
            3141.712169817681, 2601.8722089847142, 10453.30588154209, 2497.529052726244, 338.9759031674222]
r_mean = 6334.413313103378
r_std = 4037.501046550905

#Dagger Policy
d_returns = [10676.754776987436, 10654.731779936332, 10721.995014733453, 10716.626302259205, 10693.668150622354, 
            10663.160179151275, 10725.854070690499, 10692.370993527007, 10612.604647565415, 10752.66984988682, 
            10631.351589723428, 10752.932786007852, 10672.022198984572, 10685.082008038264, 10710.379528530275, 
            10758.335368122607, 10696.62074086837, 10667.322888877301, 10751.482504927502, 10622.06966859705, 
            10610.955318553932, 10738.860765657768, 10740.548563421513, 10581.552940984964, 10688.037923229691]
d_mean = 10688.719622395394
d_std = 48.905618047443674

rollouts =[i for i in range(25)]



#plt.plot(rollouts, e_returns, 'b.', label='Expert Policy',alpha=0.5)
#expert = plt.axhline(y=e_mean, color='b',alpha=0.5)

#plt.plot(rollouts, d_returns, 'g-', label='DAgger Policy',alpha=0.5)
#dagger = plt.axhline(y=d_mean, color='g',alpha=0.5)

#plt.plot(rollouts, r_returns, 'y*', label='Regular Policy',alpha=0.5)
#regular = plt.axhline(y=r_mean, color='y',alpha=0.5)

#plt.legend(loc=3)
plt.xlabel('Rollout No.')
plt.ylabel('Policy Reward')
plt.title('Policy Performance based on Rewards')


#Second Graph
frequencies = [e_std,d_std,r_std]

# In my original code I create a series and run on that,
# so for consistency I create a series from the list.
freq_series = pd.Series(frequencies)

x_labels = ['Expert', 'DAgger', 'Regular']

# Plot the figure.
ax = freq_series.plot(kind='bar')
ax.set_title('Amount Frequency')
ax.set_xlabel('Amount ($)')
ax.set_ylabel('Frequency')
ax.set_xticklabels(x_labels)

def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.1f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.


# Call the function above. All the magic happens there.
add_value_labels(ax)
plt.show()