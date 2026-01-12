import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
df = pd.read_csv('housing.csv')  
 
# 2. Ιστογράμματα  (PDF) για όλες τις 10 μεταβλητές
 
for col in df.columns:
    plt.figure(figsize=(8, 6))
    # Ελέγχουμε εάν η μεταβλητή είναι αριθμητική ή κατηγορική
    if pd.api.types.is_numeric_dtype(df[col]):
        sns.histplot(df[col], kde=True, bins=50)  # Χρήση KDE για εκτίμηση της συνάρτησης πυκνότητας πιθανότητας
        plt.title(f'Ιστόγραμμα Συχνοτήτων για το {col}')
        plt.xlabel(col)
        plt.ylabel('Συχνότητα')
    else:
        sns.countplot(x=df[col])
        plt.title(f'Διάγραμμα Κατανομής για το {col}')
        plt.xlabel(col)
        plt.ylabel('Συχνότητα')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

 
# Γράφημα 1: Longitude vs Latitude
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='longitude', y='latitude')
plt.title('Scatter Plot: Longitude vs Latitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.tight_layout()
plt.show()
# Γράφημα 2: Median Income vs Median House Value
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='median_income', y='median_house_value')
plt.title('Scatter Plot: Median Income vs Median House Value')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.grid(True)
plt.tight_layout()
plt.show()
 
# Γράφημα 1: Longitude vs Latitude, χρώμα: Median House Value
plt.figure(figsize=(8, 6))
sc = plt.scatter(df['longitude'], df['latitude'], c=df['median_house_value'], cmap='viridis', alpha=0.7)
plt.title('Scatter Plot (3 μεταβλητές):\nLongitude vs Latitude\nColor: Median House Value')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
cbar = plt.colorbar(sc)
cbar.set_label('Median House Value')
plt.grid(True)
plt.tight_layout()
plt.show()
# Γράφημα 2: Median Income vs Median House Value, χρώμα: Housing Median Age
plt.figure(figsize=(8, 6))
sc = plt.scatter(df['median_income'], df['median_house_value'], c=df['housing_median_age'], cmap='plasma', alpha=0.7)
plt.title('Scatter Plot (3 μεταβλητές):\nMedian Income vs Median House Value\nColor: Housing Median Age')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
cbar = plt.colorbar(sc)
cbar.set_label('Housing Median Age')
plt.grid(True)
plt.tight_layout()
plt.show()
# median_income vs median_house_value,χρώμα: housing_median_age ,style population_bin
df['population_bin'] = pd.cut(df['population'], bins=3, labels=['Low', 'Medium', 'High'])
# Create a scatter plot with 4 variables:
# - x: median_income
# - y: median_house_value
# - hue (color): housing_median_age
# - style (marker shape): population_bin
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x='median_income',
    y='median_house_value',
    hue='housing_median_age',      # continuous variable represented by color
    style='population_bin',        # binned variable represented by marker shape
    palette='viridis',
    s=150,                         # fixed marker size
    alpha=0.7
)
plt.title('Scatter Plot with 4 Variables:\n'
          'X: Median Income, Y: Median House Value\n'
          'Hue: Housing Median Age, Style: Population Bin')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()