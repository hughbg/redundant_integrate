"""
Statistics of group integration, 1 time
Real. Mean: -0.0007 Sigma: 0.2825 Skew: 0.0025 Kurtosis: 0.1169 Bias -0.0025
Imag. Mean: 0.0018 Sigma: 0.2834 Skew: 0.0031 Kurtosis: 0.1032 Bias 0.0063
Done histogram
Done increasing bls
Final time instegration statistics
Real. Mean: 0.0 Sigma: 0.0094 Skew: -0.0018 Kurtosis: 0.0131 Bias 0.0021
Imag. Mean: 0.0 Sigma: 0.0093 Skew: -0.0022 Kurtosis: -0.028 Bias 0.0016
Done increasing times

"""
import sys

def fut(i1, i2):
    return "("+lines[i1][i2]+", "+lines[i1+1][i2]+")"

lines = []
for i, line in enumerate(sys.stdin):
    lines.append(line.split())

s = """
\\begin#tabular%#|c|c|c|c|c|%
\\hline
 \\textbf#Statistics for 50 bls% & $\\mu$ (real, imag) &  $\\sigma$ (real, imag) & skew (real, imag) & kurt (real, imag)\\\\
\\hline
Red group (1 time) & {a} & {b} & {c} & {d} \\\\ 
\parbox#1.4inr%#\\vspace#0.01in%
Red group (averaged\\\\
over all times)% & {e} & {f}  & {g} & {h}\\\\ 
\\hline
\\end#tabular%
""".format( a=fut(1, 2), b=fut(1, 4), c=fut(1, 6), d=fut(1, 8), e=fut(6, 2), f=fut(6, 4), g=fut(6, 6), h=fut(6, 8))

s = s.replace("#", "{")
print(s.replace("%", "}"))
