import geopandas as gpd
import matplotlib.pyplot as plt

# load in the UCRB boundary
ucrb = gpd.read_file("/home/dlhogan/GitHub/Synoptic-Sublimation/01_data/geodata/Upper_Colorado_River_Basin_Boundary.json", driver="GeoJSON")
print(ucrb.head())
# plot the UCRB boundary
# fig, ax = plt.subplots()
# ucrb.boundary.plot(ax=ax, color="black")
# remove all axis labels and spines
# ax.axis("off")
# save the figure as a vector without background
# plt.savefig("/home/dlhogan/GitHub/Synoptic-Sublimation/04_products/figures/draft/ucrb_outline_vector.svg", format="svg", transparent=True)