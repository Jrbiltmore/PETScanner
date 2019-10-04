import multiprocessing as mp
import os


def run_fn(f):
    os.system('mkdir %d; cd %d;../bin/Detect2000 ../input_files/input_mult_%d.det pmt.dat' % (f, f, f))


def main():
    print("Number of processors: ", mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())
    pool.map(run_fn, [file for file in range(1000)])
    pool.close()


if __name__ == "__main__":
    main()
