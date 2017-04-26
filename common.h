#ifndef _COMMON_H
#define _COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int SigLen = 256;
const int FFTRun = 10;
const int Trials = 10;

const char short_opt[] = {'l', 'r', 't', 'u', '\0'};
const char *long_opt[] = {"--len", "--runs", "--trials", "--usage", ""};

void usage(char *argv)
{
	printf("Usage:\n");
	printf("\t%s opt opt_val\n\n", argv);
	printf(" -l    (--len)         fft_len\n");
	printf(" -r    (--runs)        fft runs\n");
	printf(" -t    (--trials)      trials\n");
}


int moreopt(char *argv)
{
	int i=0;
	while(short_opt[i]!='\0'){
		if(strcmp(argv, long_opt[i])==0){
			argv[1]=short_opt[i];
			argv[2]='\0';
			return 0;
		}
		i++;
	}
	return 1;
}

int read_opt(int argc, char **argv, int id, void *data, const char *datatype) 
{
	if(strcmp(datatype, "int")==0) {
		if(id+1 >= argc) {
			fprintf(stderr, "incomplete input");
			usage(argv[0]);
			exit(EXIT_FAILURE);
		}
		*((int*)data)=atoi(argv[id+1]);
	}

	return id+1;
}


#endif
