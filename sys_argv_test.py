import sys

a = 1

if __name__ == '__main__':
	print(sys.argv)
	print('len argv', len(sys.argv))
	if len(sys.argv)>1:
		a = sys.argv[1]
	print(a)
