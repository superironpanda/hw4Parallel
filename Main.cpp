#include <stdlib.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <cmath>

using namespace std;

int mA = 500;
int nA = 500;
int nB = 500;
vector< vector<int> > createRandomMatrix(int m, int n);
vector< vector<int> > createEmptyMatrix(int m, int n);
vector< vector<int> > serialMM(vector< vector<int> > A, vector< vector<int> > B, vector< vector<int> > C);
void TwoDMM(vector< vector<int> > A, vector< vector<int> > B, vector< vector<int> > C, int world_rank, int processor_rank);
void findDiff(vector< vector<int> > A, vector< vector<int> > B);
void OneDMM(vector< vector<int> > A, vector< vector<int> > B, vector< vector<int> >, int world_rank, int processor_rank);
void printMatrix(vector< vector<int> > matrix);
void runSerialMM(int argc, char* argv[], vector< vector<int> > A, vector< vector<int> > B, vector< vector<int> > C);
void runOneDMM(int argc, char* argv[], vector< vector<int> > A, vector< vector<int> > B, vector< vector<int> > C);
void runTwoDMM(int argc, char* argv[], vector< vector<int> > A, vector< vector<int> > B, vector< vector<int> > C);

int main(int argc, char *argv[]) {
    /*int mA = 10, nA = 10, B_ROW = nA, nB = 10; 
    int option = 2;
    if (argc > 4) {
        mA = atoi(argv[argc - 4]);
        nA = atoi(argv[argc - 3]);
        nB = atoi(argv[argc - 2]);
        option = atoi(argv[argc - 1]);
    }*/
    vector< vector<int> > A = createRandomMatrix(mA, nA);
    vector< vector<int> > B = createRandomMatrix(nA, nB);
    vector< vector<int> > C = createEmptyMatrix(mA, nB);
    runSerialMM(argc, argv, A, B, C);
    runOneDMM(argc, argv, A, B, C);
    runTwoDMM(argc, argv, A, B, C);
}

void runSerialMM(int argc, char* argv[], vector< vector<int> > A, vector< vector<int> > B, vector< vector<int> > C){

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_rank);

    // Get the rank of the process
    int processor_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &processor_rank);

    //serial matrix multiplications
    int working_unit = rand() % world_rank;
    if(processor_rank == working_unit){
        C = serialMM(A, B, C);
        //printMatrix(C);
    }

    // Finalize the MPI environment.
    //MPI_Finalize();

    
}

vector< vector<int> > createRandomMatrix(int m, int n) {
    srand(time(NULL));
    vector< vector<int> > matrix;
    for (int i = 0; i < m; i++) {
        vector<int> tmp;
        for (int j = 0; j < n; j++) {
            tmp.push_back(rand() % 10 + 1);
        }
        matrix.push_back(tmp);
    }
    return matrix;
}

vector< vector<int> > createEmptyMatrix(int m, int n){
    vector< vector<int> > matrix;
    for (int i = 0; i < m; i++) {
        vector<int> tmp;
        for (int j = 0; j < n; j++) {
            tmp.push_back(0);
        }
        matrix.push_back(tmp);
    }
    return matrix;
}

vector< vector<int> > serialMM(vector< vector<int> > A, vector< vector<int> > B, vector< vector<int> > C){
    double timeSpent = 0.0;
    clock_t begin = clock();

    for (int i = 0; i < mA; i++) {
        for (int j = 0; j < nA; j++) {
            int tmp = 0;
            for (int k = 0; k < nB; k++) {
                tmp += A[i][k] * B[k][j];   
            }
            C[i][j] = tmp;
        }
    }

    clock_t end = clock();
    timeSpent += (double)(end - begin) / CLOCKS_PER_SEC;
    cout << "Serial Times used is " << timeSpent << " seconds."<<endl;
    
    return C;
}

void OneDMM(vector< vector<int> > A, vector< vector<int> > B, vector< vector<int> > C, int world_rank, int processor_rank){
    
    int A_row_chunck = mA / (world_rank - 1);
    int processor_C[A_row_chunck][nB];
    int t[nA][mA];
    int t2[mA][nB];
    for(int i=0; i<nA; i++){
        for(int j=0;j<mA;j++){
            t[i][j]=A[i][j];
            t2[i][j] = B[i][j];
        }
    }
    if (processor_rank == 0){
        double timeSpent = 0.0;
        clock_t begin = clock();

        for(int i=1;i<world_rank;i++){
            MPI_Send(&t, mA*nA, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&t2, mA*nA, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        for(int i=1;i<world_rank;i++){
            MPI_Recv(processor_C, A_row_chunck*nB, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int j=0; j<A_row_chunck;j++){
                for(int k=0; k<nB;k++){
                    C[(i-1)*A_row_chunck+j][k] = processor_C[j][k];
                }
            }
        }
        clock_t end = clock();
        timeSpent += (double)(end - begin) / CLOCKS_PER_SEC;
        cout << "1D parallel Times used is " << timeSpent << " seconds."<<endl;

    }
    else{
        MPI_Recv(&t, nA*mA, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&t2, mA*nB, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int i=0; i<A_row_chunck;i++){
            for(int j=0;j<nB;j++){
                int part_result = 0;
                for (int k=0;k<mA;k++){
                    part_result+=t[(processor_rank-1)*A_row_chunck+i][k] * t2[k][j];
                }
                processor_C[i][j]=part_result;
            }
        }
        MPI_Send(processor_C, A_row_chunck*nB, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}

void runOneDMM(int argc, char* argv[], vector< vector<int> > A, vector< vector<int> > B, vector< vector<int> > C){
    // Initialize the MPI environment
    //MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_rank);

    int processor_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &processor_rank);

    OneDMM(A, B, C, world_rank, processor_rank);
    

    // Finalize the MPI environment.
    //MPI_Finalize();

    //printMatrix(C);
}

void printMatrix(vector< vector<int> > matrix) {
    cout << endl;
    int m = matrix.size(), n = matrix[0].size();
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << matrix[i][j] << " ";
        }

        // Newline for new row 
        cout << endl;
    }
    cout << endl;
}

void findDiff(vector< vector<int> > A, vector< vector<int> > B){
    int diff = 0;

    for (int i = 0; i < mA; i++) {
        for (int j = 0; j < nB; j++) {
            if (A[i][j] != B[i][j]) {
                diff += 1;
            }
        }
    }
    cout << "diff: " << diff << endl;
}

void TwoDMM(vector< vector<int> > A, vector< vector<int> > B, vector< vector<int> > C, int world_rank, int processor_rank){
    int tmp = (mA / int(sqrt(world_rank - 1)));
    int processor_C[tmp][nB];
    int u, jStart, jEnd, kStart, kEnd;
    int working_unit = rand() % world_rank;
    int t[nA][mA];
    int t2[mA][nB];
    for(int i=0; i<nA; i++){
        for(int j=0;j<mA;j++){
            t[i][j]=A[i][j];
            t2[i][j] = B[i][j];
        }
    }
    if(processor_rank == 0){
        double timeSpent = 0.0;
        clock_t begin = clock();
        for (int i=1; i<world_rank;i++){
            MPI_Send(&t, mA*nA, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&t2, mA*nA, MPI_INT, i, 0, MPI_COMM_WORLD);
            
        }
        
        for(int i=1; i<world_rank;i++){
            MPI_Recv(processor_C, tmp * tmp, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            u = ((i-1) / int(sqrt(world_rank-1)));
            jStart = u * tmp;
            jEnd = jStart + tmp;
            kStart = (i-1) * tmp % nB;
            kEnd = jStart + tmp;
            for (int j=jStart;j<jEnd;j++){
                for(int k=kStart;k<kEnd;k++){
                    if(i>=mA || j>=mA || k>=mA){
                        break;
                    }
                    C[j][k]=processor_C[j-jStart][k-kStart];
                }
            }
        }
        clock_t end = clock();
        timeSpent += (double)(end - begin) / CLOCKS_PER_SEC;
        cout << "2D parallel Times used is " << timeSpent << " seconds."<<endl;
        //printMatrix(C);
    }
    else{
        u = floor(((processor_rank-1) / int(sqrt(world_rank-1))));
        jStart = min(u * tmp, mA);
        jEnd = min(jStart + tmp, mA);
        kStart = min((processor_rank-1) * tmp % nB, mA);
        kEnd = min(jStart + tmp, mA);
        /*cout<<"processor: "<<processor_rank<<endl;
        cout<<"j: "<<jEnd<<endl;
        cout<<"k: "<<kEnd<<endl;
        cout<<"u: "<<u<<endl;
        cout<<"tmp: "<<tmp<<endl;*/
        MPI_Recv(&t, nA*mA, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&t2, mA*nB, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for (int j = jStart; j < jEnd; j++) {
            for (int k = kStart; k < kEnd; k++) {
                int sum = 0;
                for (int i = 0; i < nB; i++) {
                    if(i>=mA || j>=mA || k>=mA){
                        break;
                    }
                    sum += t[j][i] * t2[i][k];
                }
                if(j>=mA || k>=mA){
                        break;
                }
                processor_C[j - jStart][k - kStart] = sum;
            }
        }
        
        MPI_Send(processor_C, tmp * tmp, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

}

void runTwoDMM(int argc, char* argv[], vector< vector<int> > A, vector< vector<int> > B, vector< vector<int> > C){
    // Initialize the MPI environment
    //MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_rank);

    int processor_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &processor_rank);

    TwoDMM(A, B, C, world_rank, processor_rank);
    

    // Finalize the MPI environment.
    MPI_Finalize();

    //printMatrix(C);
}