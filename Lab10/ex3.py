import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import math

csv_file_path = 'co2_daily_mlo.csv'
daily_data = []


with open(csv_file_path, 'r') as file:
    csv_reader = csv.DictReader(file)


    for row in csv_reader:
        daily_data.append(row)
        # print(row)
monthly_data_dict = {}

for dd in daily_data:
    # print(dd['Month'])
    entry = str(dd['Year']) + "_" + str(dd["Month"])

    if entry in monthly_data_dict.keys():
        monthly_data_dict[entry].append(dd)
    else: 
        monthly_data_dict[entry] = []
        monthly_data_dict[entry].append(dd)
# print(monthly_data_dict.keys())
monthly_data = []
for key in monthly_data_dict.keys():
    first_entry = monthly_data_dict[key][0]
    year = first_entry['Year']
    month = first_entry['Month']
    co2 = np.array([float(entry['Co2']) for entry in monthly_data_dict[key]])
    avg_co2 = np.mean(co2)
    time = float(year) + float(month) / 12
    monthly_data.append([time, year, month, avg_co2])

# print(monthly_data)
time = []
co2 = []
for md in monthly_data:
    time.append(md[0])
    co2.append(md[3])

plt.plot(time, co2)
plt.show()
plt.savefig("mlo_monthly_data.pdf", format="pdf")
plt.savefig("mlo_monthly_data.png", format="png")

#Ex. b
# Calculează regresia liniară
num_months = np.arange(1, len(co2) + 1)
slope, intercept, _, _, _ = linregress(num_months, co2)

# Calculează trendul folosind regresia liniară
trend = intercept + slope * num_months
plt.plot(time, trend)
plt.show()
plt.savefig("mlo_trend.pdf", format="pdf")
plt.savefig("mlo_trend.png", format="png")

fara_contributie = co2 - trend
plt.plot(time, fara_contributie)
plt.show()
plt.savefig("fara_contributie.pdf", format="pdf")
plt.savefig("fara_contributie.png", format="png")

# Ex. c
def periodic(x, y, alpha=1, beta=1):
    return np.exp(-alpha * np.sin(beta * np.pi * (x - y)) ** 2)

last_12_months = fara_contributie[-12:]
# print(last_12_months)
cov = []
for i in range(12): 
    for x in last_12_months:
        cov.append([periodic(x, y) for j in range(12) for y in last_12_months] )
# print(np.array(cov).shape)
mean = np.zeros(12 * len(last_12_months))
# print(np.array(mean).shape)
data = np.random.multivariate_normal(mean = mean, cov = np.array(cov))

time = np.linspace(-1, 1, 144)
plt.plot(time, data)
plt.show()
plt.savefig("periodic.pdf", format="pdf")
plt.savefig("periodic.png", format="png")