/* OMP+MPI Kmeans by Tul using MPI_Allreduce and omp atomic */

#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
#include <omp.h>
#include "kmeans.h"


/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__inline static
float euclid_dist_2(int    numdims,  /* no. dimensions */
                    float *coord1,   /* [numdims] */
                    float *coord2)   /* [numdims] */
{
    int i;
    float ans=0.0;

    for (i=0; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);

    return(ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__inline static
int find_nearest_cluster(int     numClusters, /* no. clusters */
                         int     numCoords,   /* no. coordinates */
                         float  *object,      /* [numCoords] */
                         float **clusters)    /* [numClusters][numCoords] */
{
    int   index, i;
    float dist, min_dist;

    /* find the cluster id that has min distance to object */
    index    = 0;
    min_dist = euclid_dist_2(numCoords, object, clusters[0]);

    for (i=1; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, object, clusters[i]);
        /* no need square root */
        if (dist < min_dist) { /* find the min and its array index */
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}

/*----< mpi_kmeans() >-------------------------------------------------------*/
int mpi_omp_kmeans(float    **objects,     /* in: [numObjs][numCoords] */
               int        numCoords,   /* no. coordinates */
               int        numObjs,     /* no. objects */
               int        numClusters, /* no. clusters */
               float      threshold,   /* % objects change membership */
               int       *membership,  /* out: [numObjs] */
               float    **clusters,    /* out: [numClusters][numCoords] */
               MPI_Comm   comm)        /* MPI communicator */
{
    int      i, j, rank, index, loop=0, total_numObjs;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    int     *clusterSize;    /* [numClusters]: temp buffer for Allreduce */
    float    delta;          /* % of objects change their clusters */
    float    delta_tmp;
    float  **newClusters;    /* [numClusters][numCoords] */
    extern int _debug;
    extern double commu_time, bcast_time, gather_time, scatter_time, allreduce_time, reduce_time;
    double timing;

    if (_debug) MPI_Comm_rank(comm, &rank);

    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);
    clusterSize    = (int*) calloc(numClusters, sizeof(int));
    assert(clusterSize != NULL);

    newClusters    = (float**) malloc(numClusters *            sizeof(float*));
    assert(newClusters != NULL);
    newClusters[0] = (float*)  calloc(numClusters * numCoords, sizeof(float));
    assert(newClusters[0] != NULL);
    for (i=1; i<numClusters; i++)
        newClusters[i] = newClusters[i-1] + numCoords;

    timing = MPI_Wtime();
    MPI_Allreduce(&numObjs, &total_numObjs, 1, MPI_INT, MPI_SUM, comm);
    allreduce_time += MPI_Wtime() - timing;
    
    int nthreads = omp_get_max_threads();
    int N = (int) numObjs/nthreads;

    if (_debug) printf("%2d: numObjs=%d total_numObjs=%d numClusters=%d numCoords=%d\n",rank,numObjs,total_numObjs,numClusters,numCoords);

    do {
        double curT = MPI_Wtime();
        delta = 0.0;
        #pragma omp parallel for \
                    private(i,j,index) \
                    firstprivate(numObjs,numClusters,numCoords) \
                    shared(objects,clusters,membership,newClusters,newClusterSize) \
                    schedule(dynamic, N) \
                    reduction(+:delta)
        for (i=0; i<numObjs; i++) {
            /* find the array index of nestest cluster center */
            index = find_nearest_cluster(numClusters, numCoords, objects[i],
                                         clusters);

            /* if membership changes, increase delta by 1 */
            if (membership[i] != index) delta += 1.0;

            /* assign the membership to object i */
            membership[i] = index;

            /* update new cluster centers : sum of objects located within */
            newClusterSize[index]++;
            for (j=0; j<numCoords; j++)
                newClusters[index][j] += objects[i][j];
        }

        timing = MPI_Wtime();
        /* sum all data objects in newClusters */
        MPI_Allreduce(newClusters[0], clusters[0], numClusters*numCoords,
                      MPI_FLOAT, MPI_SUM, comm);
        MPI_Allreduce(newClusterSize, clusterSize, numClusters, MPI_INT,
                      MPI_SUM, comm);
        allreduce_time += MPI_Wtime() - timing;

        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (clusterSize[i] > 1)
                    clusters[i][j] /= clusterSize[i];
                newClusters[i][j] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }
        
        timing = MPI_Wtime();
        MPI_Allreduce(&delta, &delta_tmp, 1, MPI_FLOAT, MPI_SUM, comm);
        allreduce_time += MPI_Wtime() - timing;
        delta = delta_tmp / total_numObjs;

        if (_debug) {
            double maxTime;
            curT = MPI_Wtime() - curT;
            MPI_Reduce(&curT, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            if (rank == 0) printf("%2d: loop=%d time=%f sec\n",rank,loop,curT);
        }
    } while (delta > threshold && loop++ < 500);

    if (_debug && rank == 0) printf("%2d: delta=%f threshold=%f loop=%d\n",rank,delta,threshold,loop);

    commu_time = allreduce_time + reduce_time;
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);
    free(clusterSize);

    return 1;
}

