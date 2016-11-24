#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

double calculate_portion(int*);
double logistic_function(double);
void foward_propagation(double [64][100], double*, double*, double [100][20], double*);
void calc_mesh_feature(double*, int, int [64][64][100]);

#define MAX 6400


int main(){
    int moji_raw[64][64][100]; // ひらがな画像
    int i, j, k = 0;
    char line[MAX];
    char temp[10];
    int count = 0;

    // ファイルの読み込み ////////////////////////////////////////////////////
    FILE* fp = fopen("Data/hira0_00L.dat", "r");
    while( fgets(line, MAX, fp) != NULL ){ //一行ずつ読み込む
      //printf("%s", line); //debug
      //文字列を一文字ずつ抜き出して整数型に変換し配列に格納
      //64×64×100の画像とする
        for(i=0;i<=63;i++){
            strncpy(temp, line+i, 1);
            int num = atoi(temp);
            moji_raw[j][i][k] = num;
            printf("%d", moji_raw[j][i][k]);
            count += 1;
            if(count==64){
                printf("\n");
                count = 0;
            }
        }
        ++j;
        if(j==64){
           k +=1;   j=0;
        }

    }
    fclose(fp);

    // 入力値y_iの準備 ////////////////////////////////////////////////////////
    double y_i[65];
    int z=0;
    calc_mesh_feature(y_i, z, moji_raw);

    //debug
    
    for(i=0;i<=64;i++){
        printf("%f\n", y_i[i]);
    }
    

    // %%%%%%%%%%%%%%%% ここまではmainでok%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    // weight 乱数初期化[0-1]/////////////////////////////////////////////////
     double w_ji[65][100]; // 中間層の重み ユニット数は100 
     double w_kj[100][20]; // // 出力層の重み 
    // w_ji 
    for(i=0;i<=64;i++){
        for(j=0;j<=99;j++){
            w_ji[i][j] = rand() / (1.0+RAND_MAX);
        }
    }
    for(i=0;i<=99;i++){
        w_ji[64][i] = 1.0;
    }
 
    //w_kj
    for(i=0;i<=99;i++){
        for(j=0;j<=19;j++){
           w_kj[i][j] = rand() / (1.0+RAND_MAX);
        }
    }

    /*
    //debug
    for(i=0;i<=63;i++){
        for(j=0;j<=99;j++){
            printf("%f\n", w_ji[i][j]);
        }
    }
    */

    // 学習アルゴリズム //////////////////////////////////////////////////////
    double y_j[100];
    double y_k[20];
    foward_propagation(w_ji, y_i, y_j, w_kj, y_k); // equation1, 2 
    // debug
    /*
    for(i=0;i<=19;i++){
        printf("%f\n", y_k[i]);
    }
    */

    // パラメータの準備 ////////////////////////////////////////////////////// 
    double alpha = 0.01; // 安定化定数 
    double eta = 0.01;   // 学習定数

    // 教師信号y^
    double y_hat[20][20];
    for(i=0;i<=19;i++){
        for(j=0;j<=19;j++){
            if(i == j){
                y_hat[i][j] = 1.0;
            }else{
                y_hat[i][j] = 0.0;
            }
        }
    }

   // 重みの更新
   // 式5 /////////////////////////////////////////////////////////////////////
   // alpha * delta w_kj -----------------------------------------------------
   double w_before[100][20];
   for(i=0;i<=99;i++){
       for(j=0;j<=19;j++){
           w_before[i][j] = alpha * w_kj[i][j];
           //printf("%f\n", w_before[i][j]);
        }
    }

   // (y_k hat - y_k) --------------------------------------------------------
   int tt = 0;
   double w_kj_temp[20];
   for(i=0;i<=19;i++){
       w_kj_temp[i] = ( y_hat[tt][i] - y_k[i] );
       //printf("%f", w_kj_temp[i]);
    }

    // (y_k hat - y_k) * y_k -------------------------------------------------
    double w_kj_temp2 = 0.0;
    for(i=0;i<=19;i++){
        w_kj_temp2 += w_kj_temp[i] * y_k[i];
    }
    //printf("%f\n", w_kj_temp2);

    // (y_k hat - y_k) * y_k * (1 - y_k) -------------------------------------
    // ww_share は式6でも使用
    double w_kj_temp3[20];
    double ww_share[20];
    for(i=0;i<=19;i++){
        w_kj_temp3[i] =  w_kj_temp2 * ( 1 - y_k[i] );
        //printf("%f\n", w_kj_temp3[i]); //debug
    }
    for(i=0;i<=19;i++){
        ww_share[i] = w_kj_temp3[i];
        w_kj_temp3[i] = eta * w_kj_temp3[i];
        //printf("%f \t %f \n", ww_share[i], w_kj_temp3[i]);
    }

    // eta * (y_k hat - y_k) * y_k * (1 - y_k) * y_j -------------------------
    double w_kj_temp4[100][20];
    for(i=0;i<=99;i++){
        for(j=0;j<=19;j++){
            w_kj_temp4[i][j] = w_kj_temp3[j] * y_j[i];
            //printf("%f", w_kj_temp4[i][j]);
        }
        //printf("\n");
    }

    // eta * (y_k hat - y_k) * y_k * (1 - y_k) * y_j + alpha delta w_k -------
    for(i=0;i<=99;i++){
        for(j=0;j<=19;j++){
            w_kj[i][j] += w_before[i][j];
            //printf("%f", w_kj[i][j]);
        }
        //printf("\n");
    }

    // 式6 ///////////////////////////////////////////////////////////////////
    // alpha * delta w_ji  ---------------------------------------------------
    double w_ji_before[65][100];
    for(i=0;i<=64;i++){
        for(j=0;j<=99;j++){
            w_ji_before[i][j] = alpha * w_ji[i][j];
        }
    }

    // eta * y_j * (1 - y_j) -------------------------------------------------
    double w_ji_temp = 0.0;
    for(i=0;i<=99;i++){
        w_ji_temp += y_j[i] * (1 - y_j[i]);
    }
    w_ji_temp = eta * w_ji_temp;
    //printf("%f", w_ji_temp);

    // eta * y_j * (1 - y_j) * y_i -------------------------------------------
    double w_ji_temp2[65];
    for(i=0;i<=64;i++){
        w_ji_temp2[i] = w_ji_temp * y_i[i];
        //printf("%f\n", w_ji_temp2[i]);
    }

    // Σ(y_k hat - y_k) * y_k * (1 - y_k) * w_kj ----------------------------
    double w_ji_temp3[100];
    for(i=0;i<=99;i++){
        for(j=0;j<=19;j++){
            w_ji_temp3[i] += ww_share[j] * w_kj[j][i];
        }
        //printf("%f", w_ji_temp3[i]);
    }

    // eta * y_j * (1 - y_j) * yi Σ(y_k hat - y_k_ * y_k * (1 - y_k) * w_kj --
    double w_ji_temp4[100][65];
    for(i=0;i<=99;i++){
        for(j=0;j<=64;j++){
            w_ji_temp4[i][j] = w_ji_temp2[j] * w_ji_temp3[i];
        }
    }

    // eta*y_j*(1-y_j)*y_i Σ(y_k hat - y_k) y_k (1-y_k)*w_kj+alpha*deltaw_ji--
    for(i=0;i<=99;i++){
        for(j=0;j<=65;j++){
            w_ji[i][j] = w_ji_temp4[i][j] + w_ji_before[i][j];
        }
    }

/*    // 出力ユニットの平均2乗誤差 ////////////////////////////////////////////
    double diff[20];
    double k_n = 20;
    double diff_sum;    
    double least_diff; // 求めたい平均2乗誤差
    // (y_hat - y_k)^2 -------------------------------------------------------
    for(i=0;i<=19;i++){
       diff[i] = ( y_hat[tt][i] - y_k[i] ) * ( y_hat[tt][i] - y_k[i] );
    }
    for(i=0;i<=19;i++){
       diff_sum += diff[i];
    }
    diff_sum += diff_sum / k_n;
*/
    

    return 0;
}


void foward_propagation(double w_ji[64][100], double* y_i, double* y_j, double w_kj[100][20], double* y_k){
    int i, j = 0;
    int i_num = 63;
    int m_num = 99;
    int o_num = 19;
    for(i=0;i<=m_num;i++){  // equation1
        for(j=0;j<=i_num;j++){
            y_j[i] += y_i[j] * w_ji[j][i];
        }
        y_j[i] = logistic_function(y_j[i]);
    }

    for(i=0;i<=m_num;i++){
        for(j=0;j<=o_num;j++){
            y_k[i] += y_j[j] * w_kj[j][i];
        }
        y_k[i] = logistic_function(y_k[i]);
    }
}

double logistic_function(double x){
    double answer = 0.0;
    answer = 1 / (1 + exp(-1 * x)  );
    return answer;
}

double calculate_portion(int* vec){
    int i = 0;
    double sum = 0.0;
    double denominator = 64.0;
    double portion = 0.0;

    for(i=0;i<=63;i++){
        sum += vec[i];
    }
    portion = sum / denominator;
    return portion;
}


void calc_mesh_feature(double* y_i, int z, int moji_raw[64][64][100]){
    // メッシュ特徴量の計算 //////////////////////////////////////////////////
    int square_vec[64];
    int s_count = 0;
    int start[8] = {0,8,16,24,32,40,48,56};
    int l,m,p = 0;
    int i, j;

    for(l=0;l<=7;l++){
        for(m=0;m<=7;m++){
            for(i=start[l];i<=start[l]+7;i++){
                for(j=start[m];j<=start[m]+7;j++){
                    square_vec[s_count] = moji_raw[i][j][z];
                    s_count++;
                }
            }
            y_i[p] = calculate_portion(square_vec);
            s_count = 0;
            p++;
        }
    }
 
    y_i[64] = 1.0;
}





/*
void inner_product(double w[64][100], double* x,  double* y, int s, int t){ // 式1
    int i, j;
    for(i=0;i<=t;i++){
        for(j=0;j<=s;j++){
            y[i] += x[j] * w[j][i]; 
        }
        y[i] = logistic_function(y[i]);
    }
}

void inner_product2(double w[100][20], double* x,  double* y, int s, int t){ // 式2
    int i, j;
    for(i=0;i<=t;i++){
        for(j=0;j<=s;j++){
            y[i] += x[j] * w[j][i]; 
//            printf("%f\n", y[i]);
        }
        y[i] = logistic_function(y[i]);
    }
}
*/

//void inner_product(double [64][100],  double*, double*, int, int);
//void inner_product2(double [100][20], double*, double*, int, int);



/*
    // メッシュ特徴量の計算 //////////////////////////////////////////////////
    double mesh_feature[64];
    int square_vec[64];
    int s_count = 0;
    k = 0;  //外でも使っているので初期化
    int start[8] = {0,8,16,24,32,40,48,56};
    int l,m,p = 0;
    
    for(l=0;l<=7;l++){
        for(m=0;m<=7;m++){
            for(i=start[l];i<=start[l]+7;i++){
                for(j=start[m];j<=start[m]+7;j++){
                    square_vec[s_count] = moji_raw[i][j][0];
                    s_count++;
                }
            }
            mesh_feature[p] = calculate_portion(square_vec);
            s_count = 0;
            p++;
        }
    }

    // 入力値y_iの準備 ////////////////////////////////////////////////////////
    double y_i[65];
    int z=0;
    calc_mesh_feature(y_i, z, moji_raw);

    //debug
    
    for(i=0;i<=64;i++){
        printf("%f\n", y_i[i]);
    }
    
*/

