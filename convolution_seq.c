#include <stdio.h>
#include <time.h>
#include "util.h"

void normalize_output(int **img, int normalize_amount, int num_rows, int num_cols, int **output_img)
{
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            output_img[i][j] = (int)img[i][j] / normalize_amount;
        }
    }
}

int kernel_sum(int **kernel, int kernel_size)
{
    int sum = 0;
    for (int i = 0; i < kernel_size; i++)
    {
        for (int j = 0; j < kernel_size; j++)
        {
            sum += kernel[i][j];
        }
    }
    if (sum == 0)
        return 1;
    else
        return sum;
}

int pixel_operation(int **kernel, int kernel_size, int **img, int row_index, int col_index)
{
    int mac = 0;
    int half = (int)kernel_size / 2;
    int start_row = abs(row_index - half);
    int start_col = abs(col_index - half);

    for (int i = start_row; i < start_row + kernel_size; i++)
    {
        for (int j = start_col; j < start_col + kernel_size; j++)
        {
            mac += kernel[i - start_row][j - start_col] * img[i][j];
        }
    }

    return mac;
}

int **extend_edges(int **img, int num_rows, int num_cols, int extend_amount)
{
    int **extended = alloc_2d_matrix(num_rows + (extend_amount * 2), num_cols + (extend_amount * 2));
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_cols; j++)
        {
            extended[extend_amount + i][extend_amount + j] = img[i][j];
        }
    }

    for (int layer = extend_amount - 1; layer >= 0; layer--)
    {
        for (int i = layer; i < (num_rows + (extend_amount * 2)); i++)
        {
            for (int j = layer; j < (num_cols + (extend_amount * 2)); j++)
            {

                if (i >= extend_amount && i <= extend_amount + (num_rows - 1) && j < extend_amount)
                {
                    extended[i][j] = extended[i][j + 1];
                }
                if (i >= extend_amount && i <= extend_amount + (num_rows - 1) && j > extend_amount + (num_cols - 1))
                {
                    extended[i][j] = extended[i][j - 1];
                }

                if (i < extend_amount && j < extend_amount)
                {
                    extended[i][j] = extended[i + 1][j + 1];
                }

                if (i < extend_amount && j > extend_amount + (num_cols - 1))
                {
                    extended[i][j] = extended[i][j - 1];
                }

                if (i < extend_amount && j >= extend_amount && j <= extend_amount + (num_cols - 1))
                {
                    extended[i][j] = extended[i + 1][j];
                }

                if (i > extend_amount + (num_rows - 1))
                {
                    extended[i][j] = extended[i - 1][j];
                }
            }
        }
    }

    int **temp = img;
    img = extended;
    dealloc_2d_matrix(temp, num_rows, num_cols);

    return extended;
}

void convolve_image(int **kernel, int kernel_size, int **img, int num_rows, int num_cols, int **output_img)
{
    int extend_amount = (int)kernel_size / 2;
    int **ext_input = extend_edges(img, num_rows, num_cols, extend_amount);

    for (int i = extend_amount; i < extend_amount + num_rows; i++)
    {
        for (int j = extend_amount; j < extend_amount + num_cols; j++)
        {
            output_img[i - extend_amount][j - extend_amount] = pixel_operation(kernel, kernel_size, ext_input, i, j);
        }
    }

    int kSum = kernel_sum(kernel, kernel_size);
    normalize_output(output_img, kSum, num_rows, num_cols, output_img);
    dealloc_2d_matrix(ext_input, num_rows + (extend_amount * 2), num_cols + (extend_amount * 2));
}

int main(int argc, char *argv[])
{
    double serial_main = 0.0;
    clock_t begin = clock();

    if (argc != 4)
    {
        return -1;
    }

    // read in image data
    int num_rows, num_columns;
    int **matrix = read_pgm_file(argv[1], &num_rows, &num_columns);

    // read in the kernel
    int kernel_size;
    int **kernel = read_pgm_file(argv[2], &kernel_size, &kernel_size);

    // create ouput
    int **output = alloc_2d_matrix(num_rows, num_columns);

    // convolve image
    convolve_image(kernel, kernel_size, matrix, num_rows, num_columns, output);

    FILE *to = fopen(argv[3], "w");
    fprintf(to, "%d\n", num_rows);
    fprintf(to, "%d\n", num_columns);
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_columns; j++)
        {
            fprintf(to, "%d ", output[i][j]);
        }
        fprintf(to, "\n");
    }

    // deallocate image matrix and kernel matrix
    dealloc_2d_matrix(kernel, kernel_size, kernel_size);
    dealloc_2d_matrix(output, num_rows, num_columns);

    clock_t end = clock();
    serial_main += (double)(end - begin) * 1000 / CLOCKS_PER_SEC;
    printf("CONVOLUTION SERIAL\n");
    printf("Serial Program time: %f ms\n", serial_main);

    return 0;
}
