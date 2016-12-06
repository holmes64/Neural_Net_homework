#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double calculate_portion(int*); 
double sigmoid_function(double); 
double root_mean_square(double [20][20], double*, int);
void wji_init(double w_ji[16][64], double delta_wji[16][64]);
void wkj_init(double w_kj [20][16], double delta_wkj[20][16]);
void make_images( int, int, int [64][64][100] );
void init_y(int, double*);
void calc_mesh_feature(double*, int, int [64][64][20]);
void calc_mesh_feature2(double*, int, int [64][64][100]);
void teacher_data(double [20][20]);
void forward_propagation(double [16][64], double [20][16], double*, double*, double*);
void back_propagation(double [16][64], double [20][16], double[16][64], double [20][16], double [20][20], double*, double*, double*, int);
void block_pro(int , int , int , int [64][64][20], int*);
void block_pro2(int , int , int , int [64][64][100], int*);

double w_ji[16][64];                                // weight wji
double w_kj[20][16];                                // weight wkj
double delta_wji[16][64];                           // update weight wji
double delta_wkj[20][16];                           // update weight wkj
int moji_learn[64][64][20];                         // train data
int moji_matrix[64][64][100];                       // test data

int  main(){

    int count = 0;
    int i=0;
    int j=0;
    int N=0;
    int train_num = 1;                              // train iteration num
    int left = 0;                                   // data's left index
    int right = 0;                                  // data's right index
    int seikai = 0;                                 // correct num
    int true_ind = 0;                               // correct ind
    double y_hat[20][20]; teacher_data(y_hat);      // supervised data
    double rms_sum = 0.0;                           // each itertion's sum of rms
    double rms_sum_mean = 100.0;                    // mean of rms
    double y_i[64]; init_y(64, y_i);                // input layer
    double y_j[16]; init_y(16, y_j);                // middle layer
    double y_k[20]; init_y(20, y_k);                // output layer
    FILE* fp[20];                                   // file pointer

    // weight init ///////////////////////////////////////////////////////////
    wji_init(w_ji, delta_wji);
    wkj_init(w_kj, delta_wkj);
   
    while(rms_sum_mean > 0.001){
        // prepare each folder's FILE pointer ////////////////////////////////
        for(left=0;left<=1;left++){
            for(right=0;right<=9;right++){
                char F_name2[] = "L.dat";
                char ll[4] = "0";
                char rr[] = "0";
                char F_name1[20] = "Data/hira0_";
                sprintf(ll, "%d", left);
                sprintf(rr, "%d", right);
                strcat(ll, rr);
                strcat(F_name1, ll);
                strcat(F_name1, F_name2);
                fp[count] = fopen(F_name1, "r");
                count++;
            }
        }
        count=0;
        // read each character = sum is 20 (あ - と) /////////////////////////
        char line[6400];
        char temp[65];
        int i2 = 0;
        int j2 = 0;
        int k2 = 0;
        int col = 0;

        for(col=0;col<=99;col++){
            for(N=0;N<=19;N++){ 
                fseek(fp[N], ((64*col)*65), SEEK_SET);
                while( fgets(line, 1000, fp[N]) != NULL ){
                    for(i2=0;i2<=63;i2++){
                        if(j2<=63){
                            strncpy(temp, line+i2,1);
                            int num = atoi(temp);
                            moji_learn[j2][i2][k2] = num / 10;
                        }
                    }
                    j2++;
                    if(j2 == 64){
                        k2++;
                    }
                }
                j2=0;
                // mesh feature(preprocessing /////////////////////////////////
                calc_mesh_feature(y_i, N, moji_learn);
                // learning algorithm /////////////////////////////////////////
                forward_propagation(w_ji, w_kj, y_i, y_j, y_k);
                back_propagation(w_ji, w_kj, delta_wji, delta_wkj, y_hat, y_i, y_j, y_k, N);
                // calculate each rms
                rms_sum += root_mean_square(y_hat, y_k, N);
            }
            k2=0;
        }
        // calculate mean of rms /////////////////////////////////////////////
        rms_sum_mean = rms_sum / 2000.0;
        printf("epoc num : %d \t mean of rms: %f\n", train_num, rms_sum_mean);
        train_num++;
        rms_sum = 0.0;
    }
    // close file pointers ///////////////////////////////////////////////////
    for(i=0;i<=19;i++){
        fclose(fp[i]);
    }
    // identification algorithm ///////////////////////////////////////////////
    // read test images ///////////////////////////////////////////////////////
    int hoge=0;
    int sss=0;
    int ttt=0;
    double y_test[20]; init_y(20, y_test);    
    for(left=0;left<=1;left++){
        for(right=0;right<=9;right++){
            make_images(left, right, moji_matrix);
            for(N=0;N<=99;N++){
                int real_ind = 0;
                double max_val = 0.0;
                calc_mesh_feature2(y_test, N, moji_matrix);
                forward_propagation(w_ji, w_kj, y_test, y_j, y_k);
                hoge++;
                for(sss=0;sss<=19;sss++){
                        if(hoge == 4){
                            printf("%f\n", y_k[sss]);
                        }
                }
                // calculate max of array ////////////////////////////////////
                for(i=0;i<=19;i++){
                    if(y_k[i] > max_val){
                        real_ind = i;
                        max_val = y_k[i];
                    }
                }
                //printf("test result: %d \t ground truth:%d \t probabilities: %f\n", real_ind, true_ind, max_val);
                if( real_ind == true_ind){
                    seikai++;
                }
            }
            true_ind++;
        }
    }
    double accuracy = 0.0;
    accuracy = seikai / 2000.0;
    printf("identification rate:%f\n", accuracy);
    return 0; 
}

// root mean square
double root_mean_square(double y_hat[20][20], double* y_k, int Tcol){
    int i=0;
    double diff[20]; init_y(20, diff);
    double k_n = 20.0;
    double diff_sum = 0.0;

    for(i=0;i<=19;i++){
        diff[i] = ( y_hat[Tcol][i] - y_k[i] ) * (y_hat[Tcol][i] - y_k[i] );
        diff_sum += diff[i];
    }
    diff_sum /= k_n;
    return diff_sum;
}

// equation 5,6
void back_propagation(double W_ji[16][64], double w_kj[20][16], double delta_wji[16][64], double delta_wkj[20][16], double y_hat[20][20], double* y_i, double* y_j, double* y_k, int Tcol){
   int i=0;
   int j=0;

   double alpha = 0.9;
   double eta = 0.01;

   for(i=0;i<=19;i++){
       for(j=0;j<=15;j++){
           delta_wkj[i][j] = alpha * delta_wkj[i][j];
       }
   }

   double w_kj_temp[20]; init_y(20, w_kj_temp);
   double ww_share[20];  init_y(20, ww_share);
   for(i=0;i<=19;i++){
       w_kj_temp[i] = (y_hat[Tcol][i] - y_k[i]) * y_k[i] * (1 - y_k[i]);
       ww_share[i] = w_kj_temp[i];
       w_kj_temp[i] = eta * w_kj_temp[i];
   }

   double w_kj_temp3[20][16];
   for(i=0;i<=19;i++){
       for(j=0;j<=15;j++){
           w_kj_temp3[i][j] = w_kj_temp[i] * y_j[j];
           delta_wkj[i][j] += w_kj_temp3[i][j];
           w_kj[i][j] += delta_wkj[i][j];
        }
   }

   // equation6
   for(i=0;i<=15;i++){
       for(j=0;j<=63;j++){
           delta_wji[i][j] = alpha * delta_wji[i][j];
        }
   }

   double w_ji_temp[16]; init_y(16, w_ji_temp);
   for(i=0;i<=15;i++){
       w_ji_temp[i] = y_j[i] * (1.0 - y_j[i]);
       w_ji_temp[i] = eta * w_ji_temp[i];
    }

    double w_ji_temp2[16][64];
    for(i=0;i<=15;i++){
        for(j=0;j<=63;j++){
            w_ji_temp2[i][j] = w_ji_temp[i] * y_i[j];
        }
    }

    double w_ji_temp3[16]; init_y(16, w_ji_temp3);
    for(i=0;i<=15;i++){
        for(j=0;j<=19;j++){
            w_ji_temp3[i] += ww_share[j] * w_kj[j][i];
        }
    }

    double w_ji_temp4[16][64];
    for(i=0;i<=15;i++){
        for(j=0;j<=63;j++){
            w_ji_temp4[i][j] = w_ji_temp2[i][j] * w_ji_temp3[i];
            delta_wji[i][j] = w_ji_temp4[i][j] + delta_wji[i][j];
            w_ji[i][j] += delta_wji[i][j];
        }
    }
}

// equation1,2
void forward_propagation(double w_ji[16][64], double w_kj[20][16], double* y_i, double* y_j, double* y_k){
    int i=0;
    int j=0;
    int input = 63;
    int medium = 15;
    int output = 19;
    init_y(16, y_j);
    init_y(20, y_k);
    
    for(i=0;i<=medium;i++){
        for(j=0;j<=input;j++){
            y_j[i] += w_ji[i][j] * y_i[j];
        }
        y_j[i] += 1.0;
        y_j[i] = sigmoid_function(y_j[i]);
    }

    for(i=0;i<=output;i++){
        for(j=0;j<=medium;j++){
            y_k[i] += w_kj[i][j] * y_j[j];
        } y_k[i] += 1.0; 
        y_k[i] = sigmoid_function(y_k[i]);
    }
}

double calculate_portion(int* vec){
    int i=0;
    double sum=0.0;
    double denominator = 64.0;
    double portion = 0.0;
    
    for(i=0;i<=63;i++){
        sum += vec[i];
    }
    portion = sum / denominator;
    return portion;
}

void calc_mesh_feature(double* y_i, int N, int moji_matrix[64][64][20]){
    int square_vec[64];
    int s_count = 0;
    int start[8] = {0,8,16,24,32,40,48,56};
    int l=0; 
    int m=0;
    int i=0;
    int j=0;
    int p=0;

    for(i=0;i<=7;i++){
        for(j=0;j<=7;j++){
            l = start[i];
            m = start[j];
            block_pro(l, m, N, moji_matrix, square_vec);
            y_i[p] = calculate_portion(square_vec);
            p++;
        }
    }
}

void calc_mesh_feature2(double* y_i, int N, int moji_matrix[64][64][100]){
    int square_vec[64];
    int s_count = 0;
    int start[8] = {0,8,16,24,32,40,48,56};
    int l=0; 
    int m=0;
    int i=0;
    int j=0;
    int p=0;

    for(i=0;i<=7;i++){
        for(j=0;j<=7;j++){
            l = start[i];
            m = start[j];
            block_pro2(l, m, N, moji_matrix, square_vec);
            y_i[p] = calculate_portion(square_vec);
            p++;
        }
    }
}


void block_pro(int l, int m, int N, int moji_matrix[64][64][20], int* square_vec ){
    int i=0;
    int j=0;
    int p=0;
    for(i=l;i<=l+7;i++){
        for(j=m;j<=m+7;j++){
            square_vec[p] = moji_matrix[i][j][N];
            p++;
        }
    }
}

void block_pro2(int l, int m, int N, int moji_matrix[64][64][100], int* square_vec ){
    int i=0;
    int j=0;
    int p=0;
    for(i=l;i<=l+7;i++){
        for(j=m;j<=m+7;j++){
            square_vec[p] = moji_matrix[i][j][N];
            p++;
        }
    }
}

void make_images( int left, int right, int moji_matrix[64][64][100] ){
    char F_name2[] = "L.dat";
    char ll[4] = "0";
    char rr[] = "0";
    char line[6400];
    char temp[65];
    int i2 = 0;
    int j2 = 0;
    int k2 = 0;
    int count = 0;
    int sss = 0;
    char F_name1[20] = "Data/hira0_";
    sprintf(ll, "%d", left);
    sprintf(rr, "%d", right);
    strcat(ll, rr);
    strcat(F_name1, ll);
    strcat(F_name1, F_name2);
    FILE* fp2 = fopen(F_name1, "r");
    while( fgets(line, 1000, fp2) != NULL ){
        for(i2=0;i2<=63;i2++){
            strncpy(temp, line+i2,1);
            int num = atoi(temp);
            moji_matrix[j2][i2][k2] = num;
        }
        j2++;
        if(j2 == 64){
            j2 = 0;
            k2++;
        }
    }
    fclose(fp2); 
}

void wji_init( double w_ji[16][64], double delta_wji[16][64] ){
    int i, j = 0;
    for(i=0;i<=15;i++){
        for(j=0;j<=63;j++){
            w_ji[i][j] = ((double) rand() + 1.0)/((double) RAND_MAX + 2.0);
            delta_wji[i][j] = 0.0;
        }
    }

}

void wkj_init( double w_kj[20][16], double delta_wkj[20][16] ){
    int i, j = 0;
    for(i=0;i<=19;i++){
        for(j=0;j<=15;j++){
            w_kj[i][j] = ((double) rand() + 1.0)/((double) RAND_MAX + 2.0);
            delta_wkj[i][j] = 0.0;
        }
    }

}

double sigmoid_function(double x){
    double answer;
    answer = 1.0 / (1.0 + exp(-x) );
    return answer;
}

void init_y(int t, double* y){
    int i=0;
    for(i=0;i<=t-1;i++){
        y[i] = 0.0;
    }
}

void teacher_data(double y_hat[20][20]){
    int i=0;
    int j=0;
    for(i=0;i<=19;i++){
        for(j=0;j<=19;j++){
            if(i == j){
                y_hat[i][j] = 1.0;
            }else{
                y_hat[i][j] = 0.0;
            }
        }
    }
}
