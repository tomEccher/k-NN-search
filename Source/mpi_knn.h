#ifndef MPI_KNN_H_INCLUDED
#define MPI_KNN_H_INCLUDED

#include <mpi.h>

MPI_Datatype MPI_ELEMENTI;
MPI_Datatype MPI_CAMPIONE;

void create_MPI_Elementi_type(){

    int len[4]={1,1,1, 1};

    MPI_Aint disp[4], base;
    Elementi dummy;
    MPI_Get_address(&dummy, &base);
    MPI_Get_address(&dummy.x, &disp[0]);
    MPI_Get_address(&dummy.y, &disp[1]);
    MPI_Get_address(&dummy.z, &disp[2]);
    MPI_Get_address(&dummy.idx, &disp[3]);

    disp[0]=MPI_Aint_diff(disp[0], base);
    disp[1]=MPI_Aint_diff(disp[1], base);
    disp[2]=MPI_Aint_diff(disp[2], base);
    disp[3]=MPI_Aint_diff(disp[3], base);

    MPI_Datatype types[4]={MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_UNSIGNED_LONG};
    MPI_Type_create_struct(4, len, disp, types, &MPI_ELEMENTI);
    MPI_Type_commit(&MPI_ELEMENTI);

}


void create_MPI_Campione_type(){

    int len[2]={1,1};

    MPI_Aint disp[2], base;
    Campione dummy;
    MPI_Get_address(&dummy, &base);
    MPI_Get_address(&dummy.d2, &disp[0]);
    MPI_Get_address(&dummy.idx, &disp[1]);

    disp[0]=MPI_Aint_diff(disp[0], base);
    disp[1]=MPI_Aint_diff(disp[1], base);


    MPI_Datatype types[2]={MPI_FLOAT, MPI_UNSIGNED_LONG};
    MPI_Type_create_struct(2, len, disp, types, &MPI_CAMPIONE);
    MPI_Type_commit(&MPI_CAMPIONE);

}


#endif // MPI_KNN_H_INCLUDED
