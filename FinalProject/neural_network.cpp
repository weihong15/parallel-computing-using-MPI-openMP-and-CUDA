#include "utils/neural_network.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <armadillo>

#include "cublas_v2.h"
#include "gpu_func.h"
#include "mpi.h"

#define MPI_SAFE_CALL(call)                                                  \
  do {                                                                       \
    int err = call;                                                          \
    if (err != MPI_SUCCESS) {                                                \
      fprintf(stderr, "MPI error %d in file '%s' at line %i", err, __FILE__, \
              __LINE__);                                                     \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

nn_real norms(NeuralNetwork& nn) {
  nn_real norm_sum = 0;

  for (int i = 0; i < nn.num_layers; ++i) {
    norm_sum += arma::accu(arma::square(nn.W[i]));
  }

  return norm_sum;
}

/* CPU implementation.
 * Follow this code to build your GPU code.
 */

// Sigmoid activation
void sigmoid(const arma::Mat<nn_real>& mat, arma::Mat<nn_real>& mat2) {
  mat2.set_size(mat.n_rows, mat.n_cols);
  ASSERT_MAT_SAME_SIZE(mat, mat2);
  mat2 = 1 / (1 + arma::exp(-mat));
}

// Softmax activation
void softmax(const arma::Mat<nn_real>& mat, arma::Mat<nn_real>& mat2) {
  mat2.set_size(mat.n_rows, mat.n_cols);
  arma::Mat<nn_real> exp_mat = arma::exp(mat);
  arma::Mat<nn_real> sum_exp_mat = arma::sum(exp_mat, 0);
  mat2 = exp_mat / repmat(sum_exp_mat, mat.n_rows, 1);
}

// feedforward pass
void feedforward(NeuralNetwork& nn, const arma::Mat<nn_real>& X,
                 struct cache& cache) {
  cache.z.resize(2);
  cache.a.resize(2);

  // std::cout << W[0].n_rows << "\n";tw
  assert(X.n_rows == nn.W[0].n_cols);
  cache.X = X;
  int N = X.n_cols;

  arma::Mat<nn_real> z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
  cache.z[0] = z1;

  arma::Mat<nn_real> a1;
  sigmoid(z1, a1);
  cache.a[0] = a1;

  assert(a1.n_rows == nn.W[1].n_cols);
  arma::Mat<nn_real> z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
  cache.z[1] = z2;

  arma::Mat<nn_real> a2;
  softmax(z2, a2);
  cache.a[1] = cache.yc = a2;
}


// gpu feedforward pass
void gpu_feedforward(NeuralNetwork& nn, arma::Mat<nn_real>& X,
                 struct cache& cache) {
  cache.z.resize(2);
  cache.a.resize(2);

  // std::cout << W[0].n_rows << "\n";tw
  assert(X.n_rows == nn.W[0].n_cols);
  cache.X = X;
  int N = X.n_cols; //hidden
  int M = nn.H[0]; //in
  int K = nn.H[1]; //hidden layer
  int C = nn.H[2]; //out

  //arma::Mat<nn_real> z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
  arma::Mat<nn_real> z1;
	//alpha *AB +beta *C
	// nn.W[0] * X + arma::repmat(nn.b[0], 1, N)
	// alpha = 1, beta = 1, 
  z1 = arma::repmat(nn.b[0], 1, N);
  nn_real alp = 1.0;
  copy2GPUwrap(nn.W[0].memptr(),X.memptr(),z1.memptr(),&alp,&alp,z1.n_rows,z1.n_cols,X.n_rows);
  cache.z[0] = z1;

  //printf("ff step 1, size z1 %d",(int)z1.n_rows);
 
  arma::Mat<nn_real> a1;
  //sigmoid(z1, a1);
  a1.set_size(z1.n_rows, z1.n_cols);
  my_sigmoid(z1.memptr(),a1.memptr(),z1.n_rows,z1.n_cols);
  cache.a[0] = a1;

  //printf("ff step 2,size a1 %d, %d",(int)a1.n_rows,(int)a1.n_cols);

  //printf("a1 num rows %d W1 num cols %d", (int)a1.n_rows,(int)nn.W[1].n_cols);
  if(a1.n_rows != nn.W[1].n_cols)
  std::cerr << "a0 and  W1 dimension mismatch" <<(int)a1.n_rows  <<
	  "why" << (int)nn.W[1].n_cols  << std::endl;
  assert(a1.n_rows == nn.W[1].n_cols);
  //arma::Mat<nn_real> z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
  arma::Mat<nn_real> z2;
  z2 = arma::repmat(nn.b[1], 1, N);
  copy2GPUwrap(nn.W[1].memptr(),a1.memptr(),z2.memptr(),&alp,&alp,z2.n_rows,z2.n_cols,a1.n_rows);
  //
  cache.z[1] = z2;
  //printf("ff step 3, size z2 %d, %d",(int)z2.n_rows,(int)z2.n_cols);


  arma::Mat<nn_real> a2;
  //softmax(z2, a2);
  a2.set_size(z2.n_rows,z2.n_cols);
  my_softmax(z2.memptr(),a2.memptr(),z2.n_rows,z2.n_cols);
  cache.a[1] = cache.yc = a2;
  //printf("ff step 4, size a2 %d, %d",(int)a2.n_rows,(int)a2.n_cols);
}


/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::Mat<nn_real>& y, nn_real reg,
              const struct cache& bpcache, struct grads& bpgrads) {
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_cols;

  // std::cout << "backprop " << bpcache.yc << "\n";
  arma::Mat<nn_real> diff = (1.0 / N) * (bpcache.yc - y);
  bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
  bpgrads.db[1] = arma::sum(diff, 1);
  arma::Mat<nn_real> da1 = nn.W[1].t() * diff;

  arma::Mat<nn_real> dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

  bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
  bpgrads.db[0] = arma::sum(dz1, 1);
}

//gpu backprop
void gpu_backprop(NeuralNetwork& nn, const arma::Mat<nn_real>& y, nn_real reg,
              const struct cache& bpcache, struct grads& bpgrads) {
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_cols;
  int M = nn.H[0];
  int K = nn.H[1]; //hidden layer
  int C = nn.H[2];

  //std::cout << "backprop " << bpcache.yc << "\n";
  arma::Mat<nn_real> diff = (1.0 / N) * (bpcache.yc - y);
  
  //bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
  //store W[1] as dW[1], as our GEMM is in place of additive matrix
  bpgrads.dW[1] = nn.W[1];
  arma::Mat<nn_real> a0t = bpcache.a[0].t();
  arma::Mat<nn_real> dw1 = bpgrads.dW[1];
  nn_real alp = 1.0;
  // alpha*AB +beta * C //
  copy2GPUwrap(diff.memptr(),a0t.memptr(),dw1.memptr(),
		  &alp,&reg,dw1.n_rows,dw1.n_cols,a0t.n_rows);
  bpgrads.dW[1] = dw1;
  //printf("bp step 1, size dw1 %d, %d",(int)dw1.n_rows,(int)dw1.n_cols);
  //

  //maybe can do reduction on this line??
  bpgrads.db[1] = arma::sum(diff, 1);
  
  //matrix multiply, use GEMM
  //arma::Mat<nn_real> da1 = nn.W[1].t() * diff;
  arma::Mat<nn_real> w1t = nn.W[1].t();
  arma::Mat<nn_real> da1(nn.W[1].n_cols,diff.n_cols,arma::fill::zeros);
  nn_real beta = 0.0;
  copy2GPUwrap(w1t.memptr(),diff.memptr(),da1.memptr(),&alp,&beta,nn.W[1].n_cols,diff.n_cols,w1t.n_cols);
  //printf("bp step 2, size dw1 %d, %d",(int)w1t.n_rows,(int)w1t.n_cols);
  //

  //maybe can pralize this, sigmoid and 1-sigmoid
  arma::Mat<nn_real> dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

  //bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
  arma::Mat<nn_real> Xt = bpcache.X.t();
  arma::Mat<nn_real> dw0 = nn.W[0];
  copy2GPUwrap(dz1.memptr(),Xt.memptr(),dw0.memptr(),&alp,&reg,dw0.n_rows,dw0.n_cols,Xt.n_rows);
  bpgrads.dW[0] = dw0;
  //printf("bp step 3, size dw0 %d, %d",(int)dw0.n_rows,(int)dw0.n_cols);
  bpgrads.db[0] = arma::sum(dz1, 1);
  //printf("bp step 4, size dz1 %d, %d",(int)bpgrads.db[0].n_rows,(int)bpgrads.db[0].n_cols);
}



/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
nn_real loss(NeuralNetwork& nn, const arma::Mat<nn_real>& yc,
             const arma::Mat<nn_real>& y, nn_real reg) {
  int N = yc.n_cols;
  nn_real ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

  nn_real data_loss = ce_sum / N;
  nn_real reg_loss = 0.5 * reg * norms(nn);
  nn_real loss = data_loss + reg_loss;
  // std::cout << "Loss: " << loss << "\n";
  return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::Mat<nn_real>& X,
             arma::Row<nn_real>& label) {
  struct cache fcache;
  feedforward(nn, X, fcache);
  label.set_size(X.n_cols);

  for (int i = 0; i < X.n_cols; ++i) {
    arma::uword row;
    fcache.yc.col(i).max(row);
    label(i) = row;
  }
}

/*
 * Train the neural network &nn
 */
void train(NeuralNetwork& nn, const arma::Mat<nn_real>& X,
           const arma::Mat<nn_real>& y, nn_real learning_rate, nn_real reg,
           const int epochs, const int batch_size, bool grad_check,
           int print_every, int debug) {
  int N = X.n_cols;
  int iter = 0;
  int print_flag = 0;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    int num_batches = (N + batch_size - 1) / batch_size;

    for (int batch = 0; batch < num_batches; ++batch) {
	
	    //int temp_size = (batch == num_batches - 1) ? N - batch_size * batch : batch_size;
      	//the above is hard to implement, to calculate first row is a nightmare. skip
	int last_col = std::min((batch + 1) * batch_size - 1, N - 1);
      arma::Mat<nn_real> X_batch = X.cols(batch * batch_size, last_col);
      arma::Mat<nn_real> y_batch = y.cols(batch * batch_size, last_col);

      struct cache bpcache;
      feedforward(nn, X_batch, bpcache);

      struct grads bpgrads;
      backprop(nn, y_batch, reg, bpcache, bpgrads);

      if (print_every > 0 && iter % print_every == 0) {
        if (grad_check) {
          struct grads numgrads;
          numgrad(nn, X_batch, y_batch, reg, numgrads);
          assert(gradcheck(numgrads, bpgrads));
        }

        std::cout << "Loss at iteration " << iter << " of epoch " << epoch
                  << "/" << epochs << " = "
                  << loss(nn, bpcache.yc, y_batch, reg) << "\n";
      }

      // Gradient descent step
      for (int i = 0; i < nn.W.size(); ++i) {
        nn.W[i] -= learning_rate * bpgrads.dW[i];
      }

      for (int i = 0; i < nn.b.size(); ++i) {
        nn.b[i] -= learning_rate * bpgrads.db[i];
      }

      /* Debug routine runs only when debug flag is set. If print_every is zero,
         it saves for the first batch of each epoch to avoid saving too many
         large files. Note that for the first time, you have to run debug and
         serial modes together. This will run the following function and write
         out files to CPUmats folder. In the later runs (with same parameters),
         you can use just the debug flag to
         output diff b/w CPU and GPU without running CPU version */
      if (print_every <= 0) {
        print_flag = batch == 0;
      } else {
        print_flag = iter % print_every == 0;
      }

      if (debug && print_flag) {
        save_cpu_data(nn, iter);
      }

      iter++;
    }
  }
}

/*
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork& nn, const arma::Mat<nn_real>& X,
                    const arma::Mat<nn_real>& y, nn_real learning_rate,
                    std::ofstream& error_file, 
                    nn_real reg, const int epochs, const int batch_size,
                    int print_every, int debug) {
 
	printf("starting parallel training\n");
       	int rank, num_procs;
  MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
  MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  int N = (rank == 0) ? X.n_cols : 0;
  MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

  int C = (rank ==0) ?y.n_rows :0;
  MPI_SAFE_CALL(MPI_Bcast(&C, 1, MPI_INT, 0, MPI_COMM_WORLD));

  int M = (rank ==0) ?X.n_rows :0;
  MPI_SAFE_CALL(MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD));

  int print_flag = 0;

  /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
     for storing elements in a column major way. Or you can allocate your own
     array memory space and store the elements in a row major way. Remember to
     update the Armadillo matrices in NeuralNetwork &nn of rank 0 before
     returning from the function. */

  // TODO

  /* allocate memory before the iterations */
  // Data sets
  
  /* iter is a variable used to manage debugging. It increments in the inner
     loop and therefore goes from 0 to epochs*num_batches */
  int iter = 0;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    int num_batches = (N + batch_size - 1) / batch_size;
    for (int batch = 0; batch < num_batches; ++batch) {
      /*
       * Possible Implementation:
       * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
       * 2. compute each sub-batch of images' contribution to network
       * coefficient updates
       * 3. reduce the coefficient updates and broadcast to all nodes with
       * `MPI_Allreduce()'
       * 4. update local network coefficient at each node
       */
      
      // TODO
      int start_col = batch*batch_size;
      int last_col = std::min((batch+1)*batch_size-1,N-1);
      int num_pics = last_col-start_col+1;
      
      //the following 2 lines error? segmentation/out of bounds. possibly due to not defined in other nodes
      //arma::Mat<nn_real> X_batch = X.cols(start_col, last_col);
      //arma::Mat<nn_real> y_batch = y.cols(start_col, last_col);

	//https://stackoverflow.com/questions/67096347/mpi-scatter-losing-values-from-the-final-partition
      int sendcounts[num_procs]; //sendcounts[i] is number of picture procs i has
      int displs[num_procs]; //cummulative, start from which picture. //for ea batch start from 0
      int res = num_pics%num_procs; //number of procs that have 1 more entry
      int size_per_process = num_pics/num_procs; //base num for each procs
      int increment =0;

      for(int processID = 0;processID<num_procs;processID++){
          displs[processID] = increment;
          sendcounts[processID] = (processID < res) ? size_per_process + 1 : size_per_process;
          increment += sendcounts[processID]; //increment should sum up to N
      }
      
      int process_size = sendcounts[num_procs];
      //int local_numbers[process_size]; cna't use this, need to store matrix
      //int local_numbers2[process_size];
      int M =  nn.H[0];
      int L = nn.H[1];
      int C =nn.H[2];
      arma::Mat<nn_real> local_y_batch(C,sendcounts[rank]);
      arma::Mat<nn_real> local_X_batch(M,sendcounts[rank]);
      int sendcounts_X[num_procs], sendcounts_y[num_procs];
      int displs_X[num_procs], displs_y[num_procs];

      for(int processID = 0;processID<num_procs;processID++){
          sendcounts_X[processID] = sendcounts[processID]*M;
          sendcounts_y[processID] = sendcounts[processID]*C;
          displs_X[processID] = displs[processID]*M;
          displs_y[processID] = displs[processID]*C;
      }
      /*
        printf("batch %d M %d, C %d\n",batch, M,C);
	//printf("batch %d start col %d, sendcounts_X[0] %d, sendcounts_X[3]"
	for(int i =0;i<num_procs;i++){
		printf("sendcounts normal %d \n", sendcounts[i]);
		printf("i %d sendcounts %d displs_X %d \n",i,sendcounts_X[i],displs_X[i]);

	}*/
      
      
      MPI_Scatterv(X.colptr(start_col), sendcounts_X, displs_X, MPI_FP, local_X_batch.memptr(), sendcounts_X[rank], MPI_FP,
                      0, MPI_COMM_WORLD);
      MPI_Scatterv(y.colptr(start_col), sendcounts_y, displs_y, MPI_FP, local_y_batch.memptr(), sendcounts_y[rank], MPI_FP,
                      0, MPI_COMM_WORLD);

      //2. compute each sub-batch of images' contribution to network coefficient updates
      struct cache bpcache;
      gpu_feedforward(nn, local_X_batch, bpcache);
      
      struct grads bpgrads;
      gpu_backprop(nn, local_y_batch, reg, bpcache, bpgrads);
      //printf("finish second part of parall\n"); 
      // * 3. reduce the coefficient updates and broadcast to all nodes with
      // * `MPI_Allreduce()'
      arma::Mat<nn_real> dW1(bpgrads.dW[0].n_rows, bpgrads.dW[0].n_cols);
      int siz = bpgrads.dW[0].n_rows * bpgrads.dW[0].n_cols;
      MPI_SAFE_CALL(MPI_Allreduce(bpgrads.dW[0].memptr(), dW1.memptr(), siz,
                                        MPI_FP, MPI_SUM, MPI_COMM_WORLD));   
  
    //printf("third part aaaa, W reducedall alright\n"); 
      arma::Mat<nn_real> dW2(bpgrads.dW[1].n_rows, bpgrads.dW[1].n_cols);
      siz = bpgrads.dW[1].n_rows*  bpgrads.dW[1].n_cols;
      MPI_SAFE_CALL(MPI_Allreduce(bpgrads.dW[1].memptr(), dW2.memptr(), siz,
                                        MPI_FP, MPI_SUM, MPI_COMM_WORLD)); 
     
	//printf("third part a, W reducedall alright\n");

      arma::Mat<nn_real> db1(bpgrads.db[0].n_rows, bpgrads.db[0].n_cols);
      siz = bpgrads.db[0].n_rows * bpgrads.db[0].n_cols;
      MPI_SAFE_CALL(MPI_Allreduce(bpgrads.db[0].memptr(), db1.memptr(), siz,
                                        MPI_FP, MPI_SUM, MPI_COMM_WORLD));

     
	//printf("third part b, b0 reduce alright\n");
      arma::Mat<nn_real> db2(bpgrads.db[1].n_rows, bpgrads.db[1].n_cols);
      
      siz = bpgrads.db[1].n_rows * bpgrads.db[1].n_cols;
      //printf("rank %d, size %d", rank, siz);
      MPI_SAFE_CALL(MPI_Allreduce(bpgrads.db[1].memptr(), db2.memptr(), siz,
                                        MPI_FP, MPI_SUM, MPI_COMM_WORLD));

    //  printf("print third part\n");
    //  printf("size correct? rows %d, cols %d", (int)bpgrads.db[1].n_rows, (int)bpgrads.db[1].n_cols);

      //4. update local network coefficient at each node
      nn.W[0] -= learning_rate/num_procs * dW1;
            nn.W[1] -= learning_rate/num_procs * dW2;
            nn.b[0] -= learning_rate/num_procs * db1;
            nn.b[1] -= learning_rate/num_procs * db2;
     
	   // printf("print fourth part");
      /*
      for (int i = 0; i < nn.W.size(); ++i) {
        nn.W[i] -= learning_rate * bpgrads.dW[i];
      }

      for (int i = 0; i < nn.b.size(); ++i) {
        nn.b[i] -= learning_rate * bpgrads.db[i];
      }*/
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
      //                    POST-PROCESS OPTIONS                          //
      // +-*=+-*=+-*=+-*=+-*=+-*=+-*=+-*=+*-=+-*=+*-=+-*=+-*=+-*=+-*=+-*= //
//print_every=10;
        //debug = 1;
      if (print_every <= 0) {
        print_flag = batch == 0;
      } else {
        print_flag = iter % print_every == 0;
      }

      if (debug && rank == 0 && print_flag) {
        // TODO
        // Copy data back to the CPU

        /* The following debug routine assumes that you have already updated the
         arma matrices in the NeuralNetwork nn.  */
        save_gpu_error(nn, iter, error_file);
      }

      iter++;
    }
  }

  // TODO
  // Copy data back to the CPU

  // TODO
  // Free memory
}

