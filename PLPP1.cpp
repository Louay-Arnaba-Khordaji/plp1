#include <iostream>
#include <fstream>
#include <omp.h>
#include <string>
using namespace std;

const int N = 500;
const int T = 500;

double U[N];
double U_new[N];

void initialize() {
    for (int i = 0; i < N; i++) {
        U[i] = 0.0;
        U_new[i] = 0.0;
    }
    U[N / 2] = 100.0;
}

void write_snapshot(ofstream& csv, int t) {
    csv << t;
    for (int i = 0; i < N; i++) {
        csv << "," << U[i];
    }
    csv << "\n";
}

void Sequential_Update_snapshots(ofstream& csv) {

    write_snapshot(csv, 0);

    for (int t = 1; t <= T; t++) {

        for (int i = 1; i < N - 1; i++)
            U_new[i] = 0.5 * (U[i - 1] + U[i + 1]);

        for (int i = 0; i < N; i++)
            U[i] = U_new[i];

        if (t % 50 == 0)
            write_snapshot(csv, t);
    }
}

void Sequential_Update_timing() {

    for (int t = 1; t <= T; t++) {

        for (int i = 1; i < N - 1; i++)
            U_new[i] = 0.5 * (U[i - 1] + U[i + 1]);

        for (int i = 0; i < N; i++)
            U[i] = U_new[i];
    }
}

void parallel_Update_snapshots(ofstream& csv) {

    write_snapshot(csv, 0);

#pragma omp parallel
    {
        for (int t = 1; t <= T; t++) {

#pragma omp for
            for (int i = 1; i < N - 1; i++)
                U_new[i] = 0.5 * (U[i - 1] + U[i + 1]);

#pragma omp for
            for (int i = 0; i < N; i++)
                U[i] = U_new[i];

#pragma omp single
            if (t % 50 == 0)
                write_snapshot(csv, t);
        }
    }
}

void parallel_Update_timing() {

#pragma omp parallel
    {
        for (int t = 1; t <= T; t++) {

#pragma omp for
            for (int i = 1; i < N - 1; i++)
                U_new[i] = 0.5 * (U[i - 1] + U[i + 1]);

#pragma omp for
            for (int i = 0; i < N; i++)
                U[i] = U_new[i];
        }
    }
}

int main() {

    // Sequential snapshots
    ofstream csv_seq("snapshots_seq.csv");
    initialize();
    Sequential_Update_snapshots(csv_seq);
    csv_seq.close();
    cout << "Sequential snapshots saved.\n";

    // Parallel snapshots for each thread count
    int Threads[4] = { 1, 2, 4, 8 };

    for (int th : Threads) {

        string filename = "snapshots_" + to_string(th) + ".csv";
        ofstream csv(filename);

        initialize();
        omp_set_num_threads(th);

        parallel_Update_snapshots(csv);

        csv.close();

        cout << "Parallel snapshots saved for " << th << " threads.\n";
    }

    // Timing + speedup
    initialize();
    double start_seq = omp_get_wtime();
    Sequential_Update_timing();
    double end_seq = omp_get_wtime();
    double seq_time = end_seq - start_seq;

    cout << "\nSequential Time: " << seq_time << " seconds\n";

    for (int th : Threads) {

        omp_set_num_threads(th);
        initialize();

        double start = omp_get_wtime();
        parallel_Update_timing();
        double end = omp_get_wtime();

        double par_time = end - start;
        double speedup = seq_time / par_time;
        double efficiency = speedup / th;

        cout << "Threads: " << th
            << " | Time: " << par_time
            << " | Speedup: " << speedup
            << " | Efficiency: " << efficiency << endl;
    }

    system("pause");
    return 0;
}