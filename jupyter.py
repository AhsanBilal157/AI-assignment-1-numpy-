#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[2]:


import numpy as np


# 2. Create a null vector of size 10 

# In[2]:


nullarr = np.zeros(10)
print(nullarr)


# 3. Create a vector with values ranging from 10 to 49

# In[4]:


arr = np.arange(10,50)
arr


# 4. Find the shape of previous array in question 3

# In[5]:


np.shape(arr) # this is 1-d arra because in answer we also see that it only show us row numbers not columns


# 5. Print the type of the previous array in question 3

# In[7]:


type(arr)


# 6. Print the numpy version and the configuration
# 

# In[53]:


print(np.version)
print(np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[9]:


arr.ndim


# 8. Create a boolean array with all the True values

# In[4]:


arrbool = np.array([True,True,True])
print(arrbool)
vec3 = np.array(range(1, 10), dtype="bool")
vec3


# 9. Create a two dimensional array
# 
# 
# 

# In[10]:


arr2_2d = np.array([[1,2,3],[4,5,6]])
print(arr2_2d)


# 10. Create a three dimensional array
# 
# 

# In[11]:


arr2_3d = np.array([[[1,2,3]],[[4,5,6]],[[7,8,9]]]) # simply creating an 3 d array
print(arr2_3d)


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[12]:


np.flip(arr)


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[15]:


nullarr = np.zeros(10)
print(nullarr)
nullarr[5] = 1
print(nullarr)


# 13. Create a 3x3 identity matrix

# In[17]:


np.identity(3)


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[19]:


arr = np.array([1, 2, 3, 4, 5])
arr.astype('float64')


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[20]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])
arr1*arr2


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[21]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],

            [7., 2., 12.]])
comparison = arr1==arr2
comparing = comparison.all()
comparing


# 17. Extract all odd numbers from arr with values(0-9)

# In[22]:


a = np.arange(10)
a[a%2==1]


# 18. Replace all odd numbers to -1 from previous array

# In[24]:


a[a%2==1]=-1
a


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[25]:


arr = np.arange(10)
arr[5:9] =12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[31]:


arr = np.ones((5,5))
print(arr)
arr[1:4,1:4] = 0
arr


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[33]:


arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
print(arr2d)
arr2d[1:2,1:2] =12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[36]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0,0] = 64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[37]:


arr = np.arange(10).reshape(2,5)
arr[0]


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[39]:


arr = np.arange(10).reshape(2,5)
arr[1:2,1:2]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[40]:


arr = np.arange(10).reshape(2,5)
arr[:,2:3]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[41]:


arr2d = np.arange(100).reshape(10,10)
arr2d
print(np.min(arr2d))
np.max(arr2d)


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[42]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a,b)


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[80]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
arr[a==b]


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[5]:


import random

name = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  
# data = np.random.randn(7,4) # --> This gives floating point numbers
data = np.random.randint(7, size=random.randrange(1, 10))
print(data)
print(np.where(name[data] != "Will"))


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[6]:



import random

name = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  
# data = np.random.randn(7,4) # --> This gives floating point numbers
data = np.random.randint(7, size=random.randrange(1, 10))
print(data)
print(np.where((name[data] != "Joe") & (name[data] != "Will")))


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[58]:


arr = np.arange(1,16)
arr2=arr.reshape(5,3)
arr2


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[59]:


arr = np.arange(1,17)
arr2=arr.reshape(2,2,4)
arr2


# 33. Swap axes of the array you created in Question 32

# In[60]:


np.swapaxes(arr2,0,2)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[61]:


arr = np.arange(10)
arr**2


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[62]:


a=np.array([345,56,32,57,97,24,68,978,24,46,8967,24])
print(a)
b=np.array([12,57,34,79,35,24,67,80,35,678,13,4567])
print(b)
n = np.maximum(a,b)
n


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[75]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[82]:


a = np.array([1,2,3,4,5]) 
b = np.array([5,6,7,8,9])
np.delete(a,4)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[3]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
sampleArray
newColumn = np.array([[10,10,10]])
sampleArray[:,1] =10
sampleArray


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[74]:


x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])
np.dot(x,y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[73]:


arr2d = np.arange(25).reshape(5,5)
print(arr2d)
np.cumsum(arr2d)

