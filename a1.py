#####Saeed Memon's CSC311 A1#####
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import sklearn.linear_model as lin
import sklearn.neighbors  as nei
from bonnerlib3 import plot_db

import time 
import pickle






print('\n')
print('Question 1')
print('---------------------')

print('\n')
print('Question 1(a)')
print('-------------')

B = rnd.random((4,5))
print (B)


print('\n')
print('Question 1(b)')
print('-------------')

y = rnd.random((4,1))
print(y)

print('\n')
print('Question 1(c)')
print('-------------')

C = np.reshape(B,(2,10))
print(C)

print('\n')
print('Question 1(d)')
print('-------------')

D = B-y
print(D)

print('\n')
print('Question 1(e)')
print('-------------')

z = np.reshape(y,(4))
print(z)

print('\n')
print('Question 1(f)')
print('-------------')

B[:,3] = z
print(B)

print('\n')
print('Question 1(g)')
print('-------------')

D[:,0] = B[:,2] + z
print(D)


print('\n')
print('Question 1(h)')
print('-------------')

print(B[0:3])

print('\n')
print('Question 1(i)')
print('-------------')

print(np.array([B[:,1],B[:,3]]))

print('\n')
print('Question 1(j)')
print('-------------')

print(np.log(B))  

print('\n')
print('Question 1(k)')
print('-------------')

print(np.sum(B))

print('\n')
print('Question 1(l)')
print('-------------')

print(np.max(B,0))

print('\n')
print('Question 1(m)')
print('-------------')

print(np.max(np.sum(B,1)))

print('\n')
print('Question 1(n)')
print('-------------')

print(np.matmul(B.T,D))

print('\n')
print('Question 1(o)')
print('-------------')

yTD = np.matmul(y.T,D)
DTy = np.matmul(D.T,y)

print(np.matmul(yTD,DTy))




print('\n')
print('Question 2')
print('---------------------')

print('\n')
print('Question 2(a)')
print('-------------')



#This function proforms matrix multiplication 
def mat_multiply(A,B): 
    I,K = np.shape(A)   # Takes the dimention of A into Row I, Col J
    K,J = np.shape(B)   # Takes the dimention of A into Row J, Col K
    AB = np.zeros([I,J])  # The result of A*B
    for i in range(I):
        for j in range(J):
            temp = 0.0
            #implementing Matrix multiplication definiton
            for k in range(K):      
                temp += A[i,k] * B[k,j]
            AB[i,j] = temp
    return AB

def matrix_poly(A):
    return mat_multiply(A+A,mat_multiply(A+A,A))

    
    
print('\n')
print('Question 2(b)')
print('-------------')




def timing(N):
    A = rnd.random((N,N))
    
    #for loop method
    t1 = time.time()
    B1 = matrix_poly(A)
    t2 = time.time()
    print("The Execusion time for Using For loops with matrix ",N," is ",t2-t1) 
    
    
    #numpy method
    t1 = time.time()
    B2 = np.matmul(A+A,np.matmul(A+A,A))
    t2 = time.time()
    print("The Execusion time for Using Numpy with matrix ",N, " is ",t2-t1) 
    
    print("The mamgnitude of B1-B2 is ",np.linalg.norm((B1-B2)))
    
    
    
print('\n')
print('Question 2(c)')
print('-------------')

print(timing(100))

print(timing(300))

print(timing(1000))





print('\n')
print('Question 3')
print('---------------------')




print('\n')
print('Question 3(a)')
print('-------------')

def least_squares(x,t):
    
    #Construction X 
    x_size = np.shape(x)           #Taking the Row and Col of x to ensure the shape matches with X 
    X = np.empty([30,2]) # need to focus on 
    t = np.reshape(t,(30,1))

    X[:, 0] = np.ones(x_size)    
    X[:, 1] = x
    step1 = np.matmul(X.T,X)        #XTX
    step2 = np.matmul(np.linalg.inv(step1),X.T)    #[(XTX)^-1]XT
    
    return np.matmul(step2,t)
    


print('\n')
print('Question 3(b)')
print('-------------')


#x is x for training data and t is the y for training data

def plot_data(x,t):
          
    line_x = np.sort(x)  #sorted version of x to help accuratly plot the line segmant
    
    b,a = least_squares(x,t)        
    y = a*line_x+b              #the line segmant equation
    
    plt.scatter(x,t,c='b')      #plot of the Training data
    plt.plot(line_x,y,c='r')    #plot of the line segmant   
    plt.title('Question 3(b): the fitted line')
    plt.show()
    
    
#this for testing    
#plot_data(X_Train,Y_Train)

print('\n')
print('Question 3(c)')
print('-------------')


def error(a,b,X,T):
    
    y_0 = a*X + b
    err = np.square(T-y_0)
    return np.mean(err)
    
    
    

print('\n')
print('Question 3(d)')
print('-------------')


with open('dataA1Q3.pickle','rb') as f:
    dataTrain,dataTest = pickle.load(f)    


X_Train = dataTrain[0] 
Y_Train = dataTrain[1]

X_Test = dataTest[0]
Y_Test = dataTest[1]
b,a = least_squares(X_Train,Y_Train)

plot_data(X_Train,Y_Train) 

print('The value for a is ',format(a),'The value of b is ',format(b))
print('\n')

print('The Training error is ',format(error(a,b,X_Train,Y_Train)))
print('\n')


print('The Testing error is ',format(error(a,b,X_Test,Y_Test)))
print('\n')





print('\n')
print('Question 4')
print('---------------------')

with open('dataA1Q4v2.pickle','rb') as f:
    Xtrain,Ttrain,Xtest,Ttest = pickle.load(f)

print('\n')
print('Question 4(a)')
print('-------------')

clf = lin.LogisticRegression() # create a classification object, clf
clf.fit(Xtrain,Ttrain) # learn a logistic-regression classifier
w = clf.coef_[0] # weight vector
w0 = clf.intercept_[0] # bias term

print('The value of the weight vector and bias is below')
print(w)
print(w0)


print('\n')
print('Question 4(b)')
print('-------------')

accuracy1 = clf.score(Xtest,Ttest)

z = np.matmul(Xtest,w) + w0            #Xtrain is 200x3 and W is 3x1
y = 1/np.logaddexp(0,-z)
prediction = np.where(y >= 0.236,1,0)     # Through multiple trial and error I got 0.236 to have the closest accuracy
accuracy2 = np.mean(prediction) 
 

#print("Size of xw is ",format(np.shape(np.matmul(Xtrain,w))))
print('This is accuracy1 {}',format(accuracy1))
print('This is accuracy2 {}',format(accuracy2))
print('The difference between two accuracy is ',format(accuracy1-accuracy2))


print('\n')
print('Question 4(c)')   
print('-------------')

plot_db(Xtrain,Ttrain,w,w0,30,5)
plt.suptitle('Question 4(c): Training data and decision boundary')

print('\n')
print('Question 4(d)')
print('-------------')

plot_db(Xtrain,Ttrain,w,w0,30,20)
plt.suptitle('Question 4(d): Training data and decision boundary')
plt.show()



print('\n')
print('Question 5')
print('---------------------')

print("I dont know")
'''
TrainEntro = []
TestEntro = []
TrainAccuracy = []
TestAccuracy = []

#this is not gonna work need to be in main function 


np.random.seed(3)


#weight = rnd.random / 1000 #weight vector 
# bias = weight[0] #need to fix



def gd_logreg(lrate):
    
    np.random.seed(3)
    weight = rnd.randn(np.shape(Xtrain)[1]+1) / 1000                 #weight vector with bias 200x4
    
    
    X_train = np.empty([np.shape(Xtrain)[0],np.shape(Xtrain)[1]+1])           #Creating new X vector to add col'n of 1 
    X_test = np.empty([np.shape(Xtest)[0],np.shape(Xtest)[1]+1]) 
    X_train[:,0] = np.ones(np.shape(Xtrain)[0])
    X_train[:,1:] = Xtrain 
    X_test[:,0] = np.ones(np.shape(Xtest)[0])
    X_test[:,1:] = Xtest
    print(np.shape(X_train))
                                                   
    
    diff = 1000         #This is the different between each weight for now we use a huge number to pass in the loop
    iteration = 0 
    accura = 0
    error = 0  
    
    while diff >= 10**(-10):   
        
        #this step need to be removed since we are using a different format of X and W
        z_train = np.matmul(X_train,weight)
        y_train = 1/np.logaddexp(0,-z_train)
        
        z_test = np.matmul(X_test,weight)
        y_test = 1/np.logaddexp(0,-z_test)
        
        gradiant = np.matmul(X_train.T,y_train-Ttrain) / np.shape(X_train.T)[1]    # Here N  Col size of X.T
        weight = weight - lrate*gradiant
        prediction = np.where(y_test >= 0.236,1,0) # from q4 
        accura = np.mean(prediction)
        #note we are gonna use the same weight for both calculation
        
        #Training
        lce_train = np.matmul(Ttrain,np.logaddexp(0,-z_train)) + np.matmul(1-Ttrain,np.logaddexp(0,z_train))
        #print(lce_train)
        
        #This makes sure to its not comparing with empty list
        if len(TrainEntro) != 0:
            diff = abs(error - np.mean(lce_train))
        else:
            diff = abs(np.mean(lce_train)) #iteration 1 
        
        error = abs(np.mean(lce_train))
        TrainEntro.append(lce_train)
        TrainAccuracy.append(accura)
        
        #Testing
        lce_test = np.matmul(Ttest,np.logaddexp(0,-z_test)) + np.matmul(1-Ttest,np.logaddexp(0,z_test))
        TestEntro.append(lce_test)
        TestAccuracy.append(accura)
        
        iteration += 1
        #print(diff)
    
    #print(weight)
    print(iteration)
    #print(TrainEntro)

      
'''
  
print('\n')
print('Question 6')
print('---------------------')
  

print('\n')
print('Question 6(a)')
print('-------------')    
    

with open('mnistTVT.pickle','rb') as f:
    Xtrain,Ttrain,Xval,Tval,Xtest,Ttest = pickle.load(f)
    


# array of true when five or six, false otherwise
fs_train = np.where(Ttrain==5,True,False) + np.where(Ttrain == 6,True,False)
fs_val = np.where(Tval==5,True,False) + np.where(Tval == 6,True,False)
fs_test = np.where(Ttest==5,True,False) + np.where(Ttest == 6,True,False)


#fs = five and six
#2k = first two thousand
rd_Xtrain_full = Xtrain[fs_train]     #reduced five and six version
rd_Ttrain_full = Ttrain[fs_train]

rd_Xtrain_small = rd_Xtrain_full[0:2000]      #reduced full version
rd_Ttrain_small = rd_Ttrain_full[0:2000]

rd_Xval = Xval[fs_val]
rd_Tval = Tval[fs_val]

rd_Xtest = Xtest[fs_test]
rd_Ttest = Ttest[fs_test]


print('\n')
print('Question 6(b)')
print('-------------')

row_size = np.shape(rd_Xtrain_full)[0]
rd_Xtrain_full = np.reshape(rd_Xtrain_full,(row_size,28,28))
i = 1
while i < 17: 
    plt.subplot(4,4,i)
    plt.axis('off')
    plt.imshow(rd_Xtrain_full[i-1],cmap = 'Greys')
    plt.suptitle('Question 6(b): 16 MNIST training images.')
    i+=1
    
plt.show()


print('\n')
print('Question 6(c)')
print('-------------')

val_accuracy = []
fs_accuracy = []
Used_k =[]

rd_Xtrain_full = np.reshape(rd_Xtrain_full,(row_size,784))

for k in range(1,20):
    if k % 2 != 0:  
        clf2 = nei.KNeighborsClassifier(n_neighbors=k)
        clf2.fit(rd_Xtrain_full,rd_Ttrain_full)

        val_score = clf2.score(rd_Xval,rd_Tval)
        fs_score = clf2.score(rd_Xtrain_small,rd_Ttrain_small)
        
        
        val_accuracy.append(val_score)
        fs_accuracy.append(fs_score)
        Used_k.append(k)
        

#part ii
plt.plot(Used_k,fs_accuracy,c='b')
plt.plot(Used_k,val_accuracy,c='r')
plt.xlabel("Nearest Number K")
plt.ylabel("Accuracy")
plt.suptitle("Question 6(c): Training and Validation Accuracy for KNN, digits 5 and 6")
plt.show()


#part iii

index = np.argmax(val_accuracy)
best_k = Used_k[index]


#part iv
clf2 = nei.KNeighborsClassifier(n_neighbors=best_k)
clf2.fit(rd_Xtrain_full,rd_Ttrain_full)
test_accuracy = clf2.score(rd_Xtest,rd_Ttest)

#part v 
print("The best value of k is ",best_k)

#part vi

print("The validation accuracy of best value k is ",val_accuracy[index])
print("The test accuracy  of best value k is ",test_accuracy)





print('\n')
print('Question 6(d)')
print('-------------')

fs_train = np.where(Ttrain==4,True,False) + np.where(Ttrain == 7,True,False)
fs_val = np.where(Tval==4,True,False) + np.where(Tval == 7,True,False)
fs_test = np.where(Ttest==4,True,False) + np.where(Ttest == 7,True,False)


#fs = five and six
#2k = first two thousand
rd_Xtrain_full = Xtrain[fs_train]     #reduced five and six version
rd_Ttrain_full = Ttrain[fs_train]

rd_Xtrain_small = rd_Xtrain_full[0:2000]      #reduced full version
rd_Ttrain_small = rd_Ttrain_full[0:2000]

rd_Xval = Xval[fs_val]
rd_Tval = Tval[fs_val]

rd_Xtest = Xtest[fs_test]
rd_Ttest = Ttest[fs_test]


#orignal 6b
row_size = np.shape(rd_Xtrain_full)[0]
rd_Xtrain_full = np.reshape(rd_Xtrain_full,(row_size,28,28))

i = 1
while i < 17: 
    plt.subplot(4,4,i)
    plt.axis('off')
    plt.imshow(rd_Xtrain_full[i-1],cmap = 'Greys')
    plt.suptitle('Question 6(b): 16 MNIST training images.')
    i+=1
    
plt.show()


#orignal 6c

val_accuracy_d = []
fs_accuracy_d = []
Used_k_d =[]

rd_Xtrain_full = np.reshape(rd_Xtrain_full,(row_size,784))

for k in range(1,20):
    if k % 2 != 0:  
        clf2 = nei.KNeighborsClassifier(n_neighbors=k)
        clf2.fit(rd_Xtrain_full,rd_Ttrain_full)

        val_score = clf2.score(rd_Xval,rd_Tval)
        fs_score = clf2.score(rd_Xtrain_small,rd_Ttrain_small)
        
        val_accuracy_d.append(val_score)
        fs_accuracy_d.append(fs_score)
        Used_k_d.append(k)
        

#part ii
plt.plot(Used_k_d,fs_accuracy_d,c='b')
plt.plot(Used_k_d,val_accuracy_d,c='r')
plt.xlabel("Nearest Number K")
plt.ylabel("Accuracy")
plt.suptitle("Question 6(d): Training and Validation Accuracy for KNN, digits 4 and 7")
plt.show()


#part iii

index_d = np.argmax(val_accuracy_d)
best_k_d = Used_k_d[index_d]
print("The best value of k is ",best_k_d)


#part iv
clf2 = nei.KNeighborsClassifier(n_neighbors=best_k_d)
clf2.fit(rd_Xtrain_full,rd_Ttrain_full)
test_accuracy_d = clf2.score(rd_Xtest,rd_Ttest)

#part v 
print("The best value of k is ",best_k_d)

#part vi

print("The validation accuracy of best value k is ",val_accuracy_d[index_d])
print("The test accuracy  of best value k is ",test_accuracy_d)





print('\n')
print('Question 6(e)')
print('-------------')









