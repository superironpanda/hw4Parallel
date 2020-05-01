/*
void TwoDMM(vector< vector<int> > A, vector< vector<int> > B, vector< vector<int> > C, int world_rank, int processor_rank){
    int A_row_chunk = (mA / int(sqrt(world_rank - 1)));
    int processor_C[A_row_chunk][nB];
    int u, tmp, jStart, jEnd, kStart, kEnd;
    int working_unit = rand() % world_rank;
    if(processor_rank == 0){
        double timeSpent = 0.0;
        clock_t begin = clock();
        for(int i=1; i<world_rank;i++){
            MPI_Recv(processor_C, A_row_chunk * A_row_chunk, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            u = ((i-1) / int(sqrt(world_rank-1)));
            tmp = (mA / sqrt(world_rank - 1));
            jStart = u * tmp;
            jEnd = jStart + tmp;
            kStart = (i-1) * tmp % nB;
            kEnd = jStart + tmp;
            for (int j=jStart;j<jEnd;j++){
                for(int k=kStart;k<kEnd;k++){
                    C[j][k]=processor_C[j-jStart][k-kStart];
                }
            }
        }
        clock_t end = clock();
        timeSpent += (double)(end - begin) / CLOCKS_PER_SEC;
        cout << "2D parallel Times used is " << timeSpent << " seconds."<<endl;
    }
    else{
        u = ((processor_rank-1) / int(sqrt(world_rank-1)));
        tmp = (mA / sqrt(world_rank - 1));
        jStart = u * tmp;
        jEnd = jStart + tmp;
        kStart = (processor_rank-1) * tmp % nB;
        kEnd = jStart + tmp;

        for (int j = jStart; j < jEnd; j++) {
            for (int k = kStart; k < kEnd; k++) {
                int sum = 0;
                for (int i = 0; i < nB; i++) {
                    sum += A[j][i] * B[i][k];
                }
                processor_C[j - jStart][k - kStart] = sum;
            }
        }
        MPI_Send(processor_C, A_row_chunk * A_row_chunk, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

}
*/