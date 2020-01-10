import numpy as np

def distance(p1, p2): 
    return np.sum((p1 - p2)**2) 

def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    
    centers = []
    centers.append(generator.randint(x.shape[0])) 
    for k in range(n_cluster - 1):   
        dist = [] 
        for i in range(x.shape[0]): 
            d = 10000000 
            for j in range(len(centers)): 
                temp = np.sum((x[i, :] -x[centers[j]])**2) 
                d = min(d, temp) 
            dist.append(d)         
        centers.append(np.argmax(np.array(dist))) 
        dist = []
#    raise Exception(
#             'Implement get_k_means_plus_plus_center_indices function in Kmeans.py')
    # DO NOT CHANGE CODE BELOW THIS LINE
    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers




def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)
#        print('The function is Boolean:'+str(centroid_func.__name__=='get_lloyd_k_means'))
#        if (centroid_func.__name__=='get_lloyd_k_means'):
#            centroids=x[self.centers]
#        else:
#            centroids=np.array(self.centers)
        
        centroids=x[self.centers]
        J=10000000000    
        for i in range(0,self.max_iter):
            tensor=np.expand_dims(centroids,axis=1)
            tensor_sum=np.sqrt(np.sum(((x - tensor) ** 2), axis=2))
            r = np.argmin(tensor_sum, axis=0)
            Jnew=0
#            print(centroids.shape)
#            print('the diference')
#            print((x[r==1]-centroids[1]).shape)
            for k in range(0,self.n_cluster):
                Jnew+=np.sum((x[r == k] - centroids[k]) ** 2)/N
            iterations=i+1
#            print(J-Jnew)
            if abs(J-Jnew)<=self.e:
                y=r
                break
            J=Jnew
            for j in range(0,self.n_cluster):
                cluster_data=x[r == j]
                if cluster_data.size==0:
                    continue
                else:
                    centroids[j]=np.mean(cluster_data, axis=0)
            if iterations==self.max_iter:
                tensor=np.expand_dims(centroids,axis=1)
                tensor_sum=np.sqrt(np.sum(((x - tensor) ** 2), axis=2))
                y = np.argmin(tensor_sum, axis=0)
        
#        for i in range(0,self.max_iter):
#            prev_centroids=centroids
#            for k in range (0,N):
#                minv=10000000
#                for j in range(0,self.n_cluster):
#                    temp=np.asarray(centroids[j,:]).reshape((-1,))
#                    dist=np.sum((x[k]-temp))**2
#                    if (dist<minv):
#                        minv=dist
#                        cnt[j]+=1
#                        summ[j,:]+=x[k,:]
#            centroids=np.divide(summ, cnt, where=cnt!=0)
#            if np.sum((centroids-prev_centroids)/prev_centroids)*100>self.e:
#                print(np.sum((centroids-prev_centroids)/prev_centroids)*100)
#                print('iter:'+str(i))
#                break
#        y=[]
#        for k in range (0,len(x)):            
#            minv=10000000
#            index=0
#            for j in range(0,self.n_cluster):
#                dist=np.sqrt(np.sum((x[k]-centroids[j])**2))
#                if (dist<minv):
#                    minv=dist
#                    index=j
#            y.append(index)
        # DONOT CHANGE CODE ABOVE THIS LINE
#        raise Exception(
#             'Implement fit function in KMeans class')
#        y=np.asarray(y)
#        print('shape of y is:'+str(y.shape))
        # DO NOT CHANGE CODE BELOW THIS LINE
#        print('centriod shape in kmean fit:after')
#        print(centroids)
        
        return centroids, y, iterations

        


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels
        
        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)
#        if (centroid_func.__name__=='get_lloyd_k_means'):
#            centroids=x[self.centers]
#        else:
#            centroids=x[np.array(self.centers)]
        centroids=x[self.centers]    
        centroid_labels=np.zeros((centroids.shape[0]))
        J=10000000000    
        for i in range(0,self.max_iter):
            tensor=np.expand_dims(centroids,axis=1)
            tensor_sum=np.sqrt(np.sum(((x - tensor) ** 2), axis=2))
            r = np.argmin(tensor_sum, axis=0)
            Jnew=0
            for k in range(0,self.n_cluster):
                Jnew+=np.sum((x[r == k] - centroids[k]) ** 2)/N
            iterations=i+1
            if abs(J-Jnew)<=self.e:
                klabel=r
                break
            J=Jnew
            for j in range(0,self.n_cluster):
                cluster_data=x[r == j]
                if cluster_data.size==0:
                    continue
                else:
                    centroids[j]=np.mean(cluster_data, axis=0)
            if iterations==self.max_iter:
                tensor=np.expand_dims(centroids,axis=1)
                tensor_sum=np.sqrt(np.sum(((x - tensor) ** 2), axis=2))
                klabel = np.argmin(tensor_sum, axis=0)
        for k in range(0,self.n_cluster):
            ind=np.where(klabel==k)
            val,cnt=np.unique(y[ind],return_counts=True)
            centroid_labels[k]=val[np.argmax(cnt)]
        # DONOT CHANGE CODE ABOVE THIS LINE
#        raise Exception(
#             'Implement fit function in KMeansClassifier class')
        
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids
#        print(centroid_labels.shape)

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels
        
        temp=np.expand_dims(self.centroids,axis=1)
        ssum=np.sqrt(np.sum(((x - temp) ** 2), axis=2))
        labels =self.centroid_labels[np.argmin(ssum, axis=0)]
        # DONOT CHANGE CODE ABOVE THIS LINE
#        raise Exception(
#             'Implement predict function in KMeansClassifier class')
#        
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'
    
    # TODO
    # - comment/remove the exception
    # - implement the function
    new_im=np.zeros((image.shape[0]*image.shape[1],image.shape[2]))
    x=np.reshape(image,(image.shape[0]*image.shape[1],image.shape[2]))
    tensor=np.expand_dims(code_vectors,axis=1)
    tensor_sum=np.sqrt(np.sum(((x - tensor) ** 2), axis=2))
    labels = np.argmin(tensor_sum, axis=0)
    for k in range(0,code_vectors.shape[0]):
        ind=np.where(labels==k)
        new_im[ind]=code_vectors[k]
#    print(code_vectors.shape)
    new_im=np.reshape(new_im,(image.shape))
    # DONOT CHANGE CODE ABOVE THIS LINE
#    raise Exception(
#             'Implement transform_image function')
#    
    
    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im

