#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>

const int NMAX = 100000;

int a[10000], b[10000], c[10000];

int is_valid_attr(int value) {
    return 2 <= value && value <= 8;
}

int main() {
    srand(time(NULL));
    int n = 0;
    int i, j, k;
    for (i = 2 ; i < 15 ; i++) for (j = 2; j < 15 - i; j++) {
        k = 15 - i - j;
        if (is_valid_attr(i) && is_valid_attr(j) && is_valid_attr(k)) {
            a[n] = i;
            b[n] = j;
            c[n] = k;
            n++;
        }
    }
    for (i = 0; i < n; i++) {
        printf("%d %d %d\n", a[i], b[i], c[i]);
    }
    for (j = 0; j < 2; j++) {
        printf("{\n");
        for (i = 0; i < 11; i++) {
            k = rand() % n;
            printf("    {%d, %d, %d},\n", a[k], b[k], c[k]);
        }
        printf("}\n");
    }
    return 0;
}
