


'''
Pseudo code in Python of how the sensor fusion 
predict-update could look like
'''
my_filter = Filter(motion_model, measurement_model)

for i, y in enumerate(measurements): 
    if i == 0:
        pred = my_filter.predict(prior)
        post = my_filter.update(y, pred)
    else:
        pred = my_filter.predict(post)
        post = my_filter.update(y, pred)