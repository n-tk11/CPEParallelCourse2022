/* Author Tul Sorawit (modified from mpi_kmean by Wei-keng Liao)*/
/* MPI kmean using gather/scatter */


#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
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
int mpi_kmeans(float    **objects,     /* in: [numObjs][numCoords] */
               int        numCoords,   /* no. coordinates */
               int        numObjs,     /* no. objects */
               int        numClusters, /* no. clusters */
               float      threshold,   /* % objects change membership */
               int       *membership,  /* out: [numObjs] */
               float    **clusters,    /* out: [numClusters][numCoords] */
               MPI_Comm   comm)        /* MPI communicator */
{
    int      i, j, k, rank, index, loop=0, total_numObjs;
    float    delta;          /* % of objects change their clusters */
    float    delta_tmp;
    extern int _debug;
    extern double commu_time, bcast_time, gather_time, scatter_time, allreduce_time, reduce_time;
    float **allObjects;
    int* allMembership;
    double timing;
    MPI_Comm_rank(comm, &rank);


    timing = MPI_Wtime();
    MPI_Allreduce(&numObjs, &total_numObjs, 1, MPI_INT, MPI_SUM, comm);
    allreduce_time += MPI_Wtime() - timing;
 
    allObjects    = (float**) malloc(total_numObjs *             sizeof(float*));
    assert(allObjects != NULL);
    allObjects[0] = (float*)  malloc(total_numObjs * numCoords * sizeof(float));
    assert(allObjects[0] != NULL);
    for (i=1; i<total_numObjs; i++)
        allObjects[i] = allObjects[i-1] + numCoords;

    allMembership = (int*) malloc(total_numObjs * sizeof(int));
    for(i=0;i<total_numObjs;i++)
        allMembership[i] = -1;

    timing = MPI_Wtime();
    MPI_Gather(objects[0], numObjs * numCoords, MPI_FLOAT, allObjects[0], numObjs * numCoords, MPI_FLOAT, 0, comm);
    gather_time += MPI_Wtime() - timing;
    

    if (_debug) printf("%2d: numObjs=%d total_numObjs=%d numClusters=%d numCoords=%d\n",rank,numObjs,total_numObjs,numClusters,numCoords);

    do {
        double curT = MPI_Wtime();
        
        //MPI_Scatter(allMembership, numObjs, MPI_INT, membership, numObjs, MPI_INT, 0, comm);
        scatter_time += MPI_Wtime() - curT;
        delta = 0.0;
        for (i=0; i<numObjs; i++) {
            /* find the array index of nestest cluster center */
            index = find_nearest_cluster(numClusters, numCoords, objects[i],
                                         clusters);

            /* if membership changes, increase delta by 1 */
            if (membership[i] != index) delta += 1.0;

            /* assign the membership to object i */
            membership[i] = index;

        }
        
        timing = MPI_Wtime();
        MPI_Gather(membership, numObjs, MPI_INT, allMembership, numObjs, MPI_INT, 0, comm);
        gather_time += MPI_Wtime() - timing;
        
        if(rank == 0){
            float total = 0;
            int numOfpoints = 0;
            for(i=0;i<numCoords;i++){
                for(j=0;j<numClusters;j++){
                    total = 0.0;
                    numOfpoints = 0;
                    for(k=0;k<total_numObjs;k++){
                        if(allMembership[k]==j){
                            total += allObjects[k][i];
                            numOfpoints++;
                        }
                    }

                    if(numOfpoints != 0){
                        clusters[j][i] = total / numOfpoints;
                    }
                }
            }
        }

        timing = MPI_Wtime();
        MPI_Bcast(clusters[0], numClusters * numCoords, MPI_FLOAT, 0, comm);
        bcast_time += MPI_Wtime() - timing;
            
        timing = MPI_Wtime();
        MPI_Allreduce(&delta, &delta_tmp, 1, MPI_FLOAT, MPI_SUM, comm);
        allreduce_time += MPI_Wtime() - timing;

        delta = delta_tmp/total_numObjs;

        if (_debug) {
            double maxTime;
            curT = MPI_Wtime() - curT;
            MPI_Reduce(&curT, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
            if (rank == 0) printf("%2d: loop=%d time=%f sec\n",rank,loop,curT);
        }
    } while (delta > threshold && loop++ < 500);

    if (_debug && rank == 0) printf("%2d: delta=%f threshold=%f loop=%d\n",rank,delta,threshold,loop);

    commu_time = gather_time + scatter_time + allreduce_time + bcast_time;

    free(allMembership);
    free(allObjects[0]);
    free(allObjects);
    
    return 1;
}
