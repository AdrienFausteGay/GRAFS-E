# Small script to add unused but prometteur plots

# %%  Here a script to plot evolution of an indicator for all territorry, all years in a clan way. Will be VERY useful for ENA and NEA stuffs
# See the result in GRAFS-E obsidian page
all_years = []
all_gammas = []
regions_gammas = {}  # dict: region -> list of gammas in the same order as annees_disponibles

for region in regions:
    gamma_list = []
    for yr in annees_disponibles:
        model = NitrogenFlowModel(data, yr, region)
        gamma_list.append(model.gamma)
    regions_gammas[region] = gamma_list
# Convert annees_disponibles to a numpy array of numeric values (sorted if needed).
x = np.array(sorted(annees_disponibles), dtype=float)

# 2) Determine the global min/max for color normalization:
all_values = []
for region in regions:
    all_values.extend(regions_gammas[region])
all_values = np.array(all_values)
vmin, vmax = all_values.min(), all_values.max()
norm = plt.Normalize(0, 2)

# 3) Build one big list of line segments + corresponding gamma values
all_segments = []
all_colors = []

for i, region in enumerate(regions):
    # y-level for this region (e.g., region i sits at y=i)
    y_level = float(i)

    # gammas for this region in the same order as x
    y_vals = regions_gammas[region]

    # Convert to numpy array
    y_vals = np.array(y_vals, dtype=float)

    # Build line segments from x[i] -> x[i+1], all at y=y_level
    # We only create segments when there are at least 2 data points
    if len(x) > 1:
        # We make "points" like [[x0, y_level], [x1, y_level], ...]
        # Then pair them up for segments
        points = np.array([x, np.full_like(x, y_level)]).T  # shape: (len(x), 2)
        segments = np.concatenate([points[:-1, None], points[1:, None]], axis=1)

        # For color, we’ll use the gamma at the start of each segment
        # (or you could average the two endpoints)
        all_segments.extend(segments)
        all_colors.extend(y_vals[:-1])

# Convert to numpy arrays for LineCollection
all_segments = np.array(all_segments)
all_colors = np.array(all_colors)

# 4) Create the global LineCollection
lc = LineCollection(all_segments, cmap="coolwarm", norm=norm)
lc.set_array(all_colors)
lc.set_linewidth(25)

# 5) Plot
fig, ax = plt.subplots(figsize=(12, len(regions) * 0.4))
ax.add_collection(lc)

# X-limits = min, max year
ax.set_xlim(x.min(), x.max())

# Y-limits = from -0.5 up to (#regions - 0.5)
ax.set_ylim(-0.5, len(regions) - 0.5)

# region labels
ax.set_yticks(range(len(regions)))
ax.set_yticklabels(regions)

# colorbar
cbar = plt.colorbar(lc, ax=ax)
cbar.set_label("Gamma")

plt.xlabel("Year")
plt.title("1D Continuous Heatmap of Gamma by Region over Year")
plt.tight_layout()
plt.show()
