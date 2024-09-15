from matplotlib.colors import LinearSegmentedColormap


# Define custom colormaps
cmap_yrbwpetrel = \
    LinearSegmentedColormap.from_list('name', ['yellow', 'red', 'black',
                                               'grey', 'white'])

cmap_amplitudepkdsg = \
    LinearSegmentedColormap.from_list('name', ['#33ffff', '#33adff', '#0000ff',
                                               '#666666', '#d9d9d9', '#805500',
                                               '#ff6600', '#ffdb4d', '#ffff00'])
cmap_amplitudepkdsg_r = \
    LinearSegmentedColormap.from_list('name',
                                      list(reversed(['#33ffff', '#33adff',
                                                     '#0000ff', '#666666',
                                                     '#d9d9d9', '#805500',
                                                     '#ff6600', '#ffdb4d',
                                                     '#ffff00'])))

cmap_bluorange = LinearSegmentedColormap.from_list('name',
                                                   list(reversed(['#004d99',
                                                                  '#ffffff',
                                                                  '#ff6600'])))

cmap_bluorange_r = LinearSegmentedColormap.from_list('name',
                                                   list(reversed(['#ff6600',
                                                                  '#ffffff',
                                                                  '#004d99'])))

cmaps = {'yrbwpetrel': cmap_yrbwpetrel,
         'amplitudepkdsg': cmap_amplitudepkdsg,
         'amplitudepkdsg_r': cmap_amplitudepkdsg_r,
         'bluorange': cmap_bluorange,
         'bluorange_r': cmap_bluorange_r,
         }