import numpy as np
from scipy import spatial
from collections import Counter
import pickle

# Build the grain structure
class grain_structure:
    
    def __init__(self, cells, size, seeds):
        
        # This will find a number of random points in the sample space from which grains will be grown
        grain_points = (seeds/size)*cells
        
        # Create a list of grain centres and GrainIDs, the grain points are also 
        # added around the edge to create a padding. 
        grain_points = np.c_[grain_points,range(0,len(grain_points))]
        
        def shift_centres(sample,x_exent,y_extend,z_extend,cells):
            sample[:,0] = sample[:,0] + cells[0]*x_exent
            sample[:,1] = sample[:,1] + cells[1]*y_extend
            sample[:,2] = sample[:,2] + cells[2]*z_extend
            return sample
        
        # Add the points around the unit so that the grains are periodic
        grain_centres = grain_points
        for x_extend in range(-1,2):
            for y_extend in range(-1,2):
                for z_extend in range(-1,2):
                    if x_extend != 0 or y_extend != 0 or z_extend != 0:
                        grain_centres = np.r_[grain_centres,shift_centres(grain_points,x_extend,
                                                                          y_extend,z_extend,cells)]
        
        # Build the matrix with grain IDs
        self.sample_grains = np.array([[[grain_centres[np.argmin(spatial.distance.cdist(grain_centres[:,0:3],
                                   np.array([[i,j,k]]))),3] + 1 for k in range(cells[2])] for j in range(cells[1])] 
                                    for i in range(cells[0])])
        
        def neighbor_counting(i,j,k,sample_grains):
            grain_IDs = sample_grains[i-1:i+2,j-1:j+2,k-1:k+2]
            if len(np.unique(grain_IDs)) == 0:
                print('woops')
            return len(np.unique(grain_IDs))
        
        # create a padded version of the sample_grains for neighbour counting
        sample_grains_padded = np.pad(self.sample_grains, 1, mode='edge')
        
        # Build a matrix containing the number of different nearest neighbours
        self.neighbor_count = np.array([[[neighbor_counting(i+1,j+1,k+1,sample_grains_padded) 
                                    for k in range(cells[2])] for j in range(cells[1])] 
                                    for i in range(cells[0])])
                             

# Add particles with a given distribution of size and position
class particle_distribution:
    
    def __init__(self):
        self = []
    
    def fill_particles(self,sample_grains,neighbor_count,cells,particle_volume_fraction, 
                       particle_average_radius,**kwargs):
        # The particles will fill with a given probability first onto the triple
        # points then the grain boundaries then the grains themselves. The probability
        # of the particle being put in any one area reduces as more particles are put in 
        # that area.
        
        # make a copy of the grain structure
        self.sample_grains = sample_grains.copy()
            
        # unravel the matrix
        neighbor_count_unravel = np.ravel(neighbor_count.copy())
        
        # make an array of indicies of the unravelled matrix
        I = np.arange(0,len(neighbor_count_unravel))
        
        # Create a matrix version of I
        I_matrix = np.reshape(I, np.shape(neighbor_count))
        
        # Create a row of the indicies in 1D
        neighbor_count_unravel_index = np.unravel_index(I,shape=np.shape(neighbor_count))
        
        # To start with some assumed proabilities that can be changed by adding the 
        # variable 'position_probs'
        new_probs = kwargs.get('position_probs', None)
        if new_probs!= None:
            probs_dict = new_probs
        else:
            probs_dict = {1:1}
            
        # If there are areas with a higher neighbour count just count them as 
        # the same as the lowest defined
        max_key = max(probs_dict, key=lambda key: probs_dict[key]) # the highest count with an assocaited probability
        for i in range(len(neighbor_count_unravel)):
            if neighbor_count_unravel[i] > max_key:
                neighbor_count_unravel[i] = max_key
                
        # From the overall chance of the particle being places on a location find the chance
        # for any given location with the same neighbour count
        neighbour_number_counts = Counter(neighbor_count_unravel)
        for key in neighbour_number_counts:
            probs_dict[key] = probs_dict[key]/neighbour_number_counts[key]
        
        # Make a list of the associated probabilities for each location
        location_probabilities = np.vectorize(probs_dict.get)(neighbor_count_unravel)
        
        # normalize to make sure the probailities sum to 1
        location_probabilities /= sum(location_probabilities)
        
        # Set the dynamic variables
        updated_location_probabilities = location_probabilities
        
        # This function is used to return all the neighbouring values in a matrix with a given square size r
        def find_neighbours(matrix,point,r):
            point_location = np.where(matrix == point)
            neighbours = np.array([[matrix[i][j][k] if i >= 0 and i < cells[0] and 
                                    j >= 0 and j < cells[1] and k >= 0 and k < cells[2] else 0
                for k in range(point_location[2][0]-1-r, point_location[2][0]+r)]
                for j in range(point_location[1][0]-1-r, point_location[1][0]+r)
                for i in range(point_location[0][0]-1-r, point_location[0][0]+r)])
            
            # This produces a list of lists of the rows, unravel this and remove zero values that 
            # occour on the edges
            neighbours = [item for sublist in neighbours for item in sublist]
            neighbours = [i for i in neighbours if i != 0]
            return neighbours
        
        # This function states how the probabilities will change when a particle is placed in a
        # given location
        def update_probabilities(location_probabilities,location,I_matrix,particle_radius):
            
            # Find the neighbours that lie within the radius of the particle
            neighbours_inner = find_neighbours(I_matrix,location,2*(particle_radius-1))
            
            # Find the neighbours in a radius of 2 from the particle
            neighbours_outer = find_neighbours(I_matrix,location,2*(particle_radius -1) + 2)
            
            # Reduce the probaility that another particle is placed within the radius of the particle to 0
            for neighbor in neighbours_inner:
                location_probabilities[neighbor] = 0
            
            # reduce the surrounding probabilities by half
            for neighbor in neighbours_outer:
                location_probabilities[neighbor] = location_probabilities[neighbor]/2
            
            # normalise the probabilities again
            location_probabilities /= sum(location_probabilities)
            
            return location_probabilities
        
        # This is used to define the area over which the particles exist
        def define_particle_area(matrix,centre,r):
            # Extract the local area
            local_area = find_neighbours(I_matrix,location,particle_radius)
            # make a copy to operate on later 
            particle_area = local_area.copy()
            # find the coordinates of the central point
            central_coords = np.where(matrix == centre)
            
            for point in local_area:
                # find the coordinates of each point
                point_coords = np.where(matrix == point)
                # determine if the point lies within a circle of radius r fromthe centre
                if (central_coords[0][0]-point_coords[0][0])**2 + (central_coords[1][0]-point_coords[1][0])**2 + (central_coords[2][0]-point_coords[2][0])**2> r**2:
                    particle_area.remove(point)
                    
            return particle_area
                
        
        # loop over the particles and keep adding them until the volume fraction is reached
        particle_density = 0
        i = 0
        while particle_density < particle_volume_fraction:
            
            # For now set the particle radius to be equal to the average
            particle_radius = particle_average_radius
            
            # Chose the location
            location = np.random.choice(I,1,p=updated_location_probabilities)
            
            # When a particle is placed on a location the probability of another particle
            # being placed on the same location falls to zero and the probabilities of the 
            # surrounding area also fall currently a using in a 4x4 matrix
            updated_location_probabilities = update_probabilities(updated_location_probabilities,location,
                                                                  I_matrix,particle_radius)
            
            # Place the particles on the grains, give them a grain value of 0
            # The particle is defined by all points that fall within the particle radius
            points_within_particle = define_particle_area(I_matrix,location,particle_radius-1)
            for point in points_within_particle:
                self.sample_grains[int(neighbor_count_unravel_index[0][point])][int(neighbor_count_unravel_index[1][point])][int(neighbor_count_unravel_index[2][point])] = 0
                
            # Find the particle density in the sample
            particle_density = np.count_nonzero(self.sample_grains == 0)/np.prod(cells)
            i += 1 # increase the iteration
            
            # in case things go very wrong
            if i > np.prod(cells):
                break
        
from damask import Grid

# input arguments
n_grains = 5                                 # Number of grains
cells = np.array([2, 256, 256])                   # Grid resolution of RVE
size = np.array([0.02, 0.64, 0.64])                # Physical size of RVE
seeds = size*np.random.rand(n_grains,3)      # Position of grain centers (random)

# If making a grain structure from scratch
sample = grain_structure(cells, size, seeds) # form grain geometry on which to place particles
pickle.dump( sample, open( "sample_grain_structure_temp.p", "wb" ) ) # save the structure
# If loading it in 
#sample = pickle.load( open( "sample_grain_structure.p", "rb" ) )

####### Test cases ###############################################
# 1. No particles
material_array = sample.sample_grains
g = Grid(material_array,size) 
g.save(fname='testcase_no_particles.vtr')

# To keep things neat
def condensed_random_distribution(name):
  particle_distribution_temp = particle_distribution()
  particle_distribution_temp.fill_particles(sample.sample_grains,sample.neighbor_count,cells,
                                     particle_vol_fraction,particle_average_radius)
  g = Grid(particle_distribution_temp.sample_grains,size) 
  g.save(fname=name)

# 2. some particles with random distribution, average radii and mid particle density 
particle_vol_fraction = 0.05
particle_average_radius = 8
condensed_random_distribution('testcase_standard.vtr')
             
# 3. High particle density
particle_vol_fraction = 0.1
particle_average_radius = 8
condensed_random_distribution('testcase_high_particle_density.vtr')

# 4. Low particle density
particle_vol_fraction = 0.01
particle_average_radius = 8
condensed_random_distribution('testcase_low_particle_density.vtr')
                                     
# 5. Small radius particle
particle_vol_fraction = 0.05
particle_average_radius = 6
condensed_random_distribution('testcase_small_radius.vtr')

# 6 big radius particle
particle_vol_fraction = 0.05
particle_average_radius = 10
condensed_random_distribution('testcase_big_radius.vtr')

# to keep things neat - this version includes the command to change the placing of particles
def condensed_random_distribution(name):
  particle_distribution_temp = particle_distribution()
  particle_distribution_temp.fill_particles(sample.sample_grains,sample.neighbor_count,cells,
                                     particle_vol_fraction,particle_average_radius,position_probs=distribution_probability)
  g = Grid(particle_distribution_temp.sample_grains,size) 
  g.save(fname=name)

# 7 grain boundary clustering
particle_vol_fraction = 0.05
particle_average_radius = 8
distribution_probability = {1:0.5,2:0.25,3:0.25}
condensed_random_distribution('testcase_weak_boundary_clustering_1.vtr')
