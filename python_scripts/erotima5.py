from scipy.stats import ttest_ind
from create_dataset import create_dataset,create_X_Y
from classes import KNN

df_data=create_dataset()
X,Y=create_X_Y(df_data)

target_class = df_data[df_data['Class'] == 1]
# Combine values of the other five classes
other_classes = df_data[df_data['Class'] != 1]
features = list(df_data.columns)
features.remove('Case #')
features.remove('Class')
#print(features)
p_values=[]

for feature in features:
    t_statistic, p_value = ttest_ind(target_class[feature], other_classes[feature])
    p_values.append([feature,p_value])

sorted_p_value_list = sorted(p_values, key=lambda x: x[1])
sorted_p_value_list = [item[0] for item in sorted_p_value_list]
sorted_p_value_list=sorted_p_value_list[:4]
print('\n')
print('ΕΡΩΤΗΜΑ 5)')
print('\n\n')
print('THIS IS THE LIST WITH THE 4 MORE IMPORTANT FEATURES')
print(sorted_p_value_list)
print('\n\n')
new_dataset = df_data.drop(sorted_p_value_list, axis=1)
Xnew,Ynew = create_X_Y(new_dataset)
knn_new=KNN(Xnew,Ynew)
knn_new.best_EXCE()


print("(ΕΡΩΤΗΜΑ 5)")
print("\n\n")

