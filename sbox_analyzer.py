import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict
import io
import base64

# Reuse the SBoxAnalyzer class from previous code
class SBoxAnalyzer:
    def __init__(self, sbox: List[int], n: int, m: int):
        self.sbox = sbox
        self.n = n
        self.m = m
        
    @staticmethod
    def hamming_weight(x: int) -> int:
        return bin(x).count('1')
    
    def truth_table(self) -> np.ndarray:
        table = []
        for output_bit in range(self.m):
            column = [(self.sbox[x] >> output_bit) & 1 for x in range(2**self.n)]
            table.append(column)
        return np.array(table)
    
    @staticmethod
    def walsh_transform(f: List[int]) -> np.ndarray:
        n = len(f).bit_length() - 1
        W = np.array(f) * 2 - 1
        for i in range(n):
            step = 2**(i + 1)
            for j in range(0, len(f), step):
                for k in range(2**i):
                    idx1, idx2 = j + k, j + k + 2**i
                    W[idx1], W[idx2] = W[idx1] + W[idx2], W[idx1] - W[idx2]
        return W
    
    def nonlinearity(self) -> float:
        table = self.truth_table()
        min_distance = float('inf')
        for column in table:
            W = self.walsh_transform(column)
            distance = 2**(self.n - 1) - np.max(np.abs(W)) / 2
            min_distance = min(min_distance, distance)
        return min_distance
    
    def sac(self) -> float:
        total = 0
        for i in range(2**self.n):
            original = self.sbox[i]
            for bit in range(self.n):
                flipped_input = i ^ (1 << bit)
                diff = original ^ self.sbox[flipped_input]
                total += self.hamming_weight(diff)
        return total / (self.n * 2**self.n * self.n)
    
    def bic_nl(self) -> float:
        table = self.truth_table()
        max_nl = 0
        for column in table:
            W = self.walsh_transform(column)
            nl = 2**(self.n - 1) - np.max(np.abs(W)) / 2
            max_nl = max(max_nl, nl)
        return max_nl
    
    def calculate_bic_sac(self) -> float:
        bit_length = 8
        total_pairs = 0
        total_independence = 0
        
        for i in range(bit_length):
            for j in range(i + 1, bit_length):
                independence_sum = self._calculate_pair_independence(i, j, bit_length)
                total_independence += independence_sum / (len(self.sbox) * bit_length)
                total_pairs += 1
        
        return round(total_independence / total_pairs, 5)
    
    def _calculate_pair_independence(self, i: int, j: int, bit_length: int) -> int:
        independence_sum = 0
        for x in range(len(self.sbox)):
            for bit_to_flip in range(bit_length):
                flipped_x = x ^ (1 << bit_to_flip)
                y1, y2 = self.sbox[x], self.sbox[flipped_x]
                
                b1_i, b1_j = (y1 >> i) & 1, (y1 >> j) & 1
                b2_i, b2_j = (y2 >> i) & 1, (y2 >> j) & 1
                
                independence_sum += (b1_i ^ b2_i) ^ (b1_j ^ b2_j)
        return independence_sum
    
    def lap(self) -> float:
        max_bias = 0
        for a in range(1, 2**self.n):
            for b in range(1, 2**self.n):
                bias = sum(1 for x in range(2**self.n)
                          if (self.hamming_weight(x & a) % 2) == 
                             (self.hamming_weight(self.sbox[x] & b) % 2))
                bias = abs(bias - 2**(self.n - 1)) / 2**self.n
                max_bias = max(max_bias, bias)
        return max_bias
    
    def dap(self) -> float:
        max_diff_prob = 0
        for dx in range(1, 2**self.n):
            for dy in range(1, 2**self.n):
                count = sum(1 for x in range(2**self.n)
                          if self.sbox[x ^ dx] ^ self.sbox[x] == dy)
                max_diff_prob = max(max_diff_prob, count / 2**self.n)
        return max_diff_prob

def create_download_link(df: pd.DataFrame, filename: str) -> str:
    """Create a download link for dataframe"""
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    b64 = base64.b64encode(excel_buffer.read()).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel file</a>'

def main():
    st.set_page_config(page_title="S-Box Analyzer", layout="wide")
    
    # Title and description
    st.title("📊 Group 3 GUI S-Box Cryptographic Analysis Tools")
    st.markdown("""
    This tool analyzes various cryptographic properties of an S-Box:
    * **NL** (Nonlinearity)
    * **SAC** (Strict Avalanche Criterion)
    * **BIC-NL** (Bit Independence Criterion - Nonlinearity)
    * **BIC-SAC** (Bit Independence Criterion - SAC)
    * **LAP** (Linear Approximation Probability)
    * **DAP** (Differential Approximation Probability)
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload S-Box Excel File", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Baca S-Box dari Excel dengan validasi
            df = pd.read_excel(uploaded_file)
            
            # Validasi format data
            if df.empty:
                st.error("File Excel kosong. Mohon upload file dengan data S-Box yang valid.")
                return
                
            # Flatten dan validasi nilai S-Box
            sbox_values = df.values.flatten().tolist()
            
            # Validasi jumlah nilai
            if len(sbox_values) != 256:  # S-Box 8x8 harus memiliki 256 nilai
                st.error("S-Box harus memiliki 256 nilai (untuk S-Box 8x8). File Anda berisi " + 
                        f"{len(sbox_values)} nilai.")
                return
                
            # Validasi range nilai
            if not all(0 <= x < 256 for x in sbox_values):
                st.error("Semua nilai dalam S-Box harus berada dalam range 0-255.")
                return
            
            # Display S-Box
            st.subheader("Imported S-Box")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(df)
            with col2:
                st.info(f"S-Box size: {len(sbox_values)} values")
            
            # Analysis options
            st.subheader("Select Analysis Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                nl_check = st.checkbox("Nonlinearity (NL)", value=True)
                sac_check = st.checkbox("SAC", value=True)
            with col2:
                bic_nl_check = st.checkbox("BIC-NL", value=True)
                bic_sac_check = st.checkbox("BIC-SAC", value=True)
            with col3:
                lap_check = st.checkbox("LAP", value=True)
                dap_check = st.checkbox("DAP", value=True)
            
            if st.button("Analyze S-Box", type="primary"):
                with st.spinner("Analyzing S-Box..."):
                    # Initialize analyzer
                    analyzer = SBoxAnalyzer(sbox_values, n=8, m=8)
                    
                    # Collect results based on selected options
                    results = {}
                    if nl_check:
                        results["NL"] = int(analyzer.nonlinearity())
                    if sac_check:
                        results["SAC"] = round(analyzer.sac(), 5)
                    if bic_nl_check:
                        results["BIC-NL"] = int(analyzer.bic_nl())
                    if bic_sac_check:
                        results["BIC-SAC"] = analyzer.calculate_bic_sac()
                    if lap_check:
                        results["LAP"] = round(analyzer.lap(), 5)
                    if dap_check:
                        results["DAP"] = round(analyzer.dap(), 6)
                    
                    # Display results
                    st.subheader("Analysis Results")
                    results_df = pd.DataFrame([results])
                    st.dataframe(results_df)
                    
                    # Create download button for results
                    excel_filename = "sbox_analysis_results.xlsx"
                    st.markdown(create_download_link(results_df, excel_filename), unsafe_allow_html=True)
                    
                    # Show success message
                    st.success("Analysis completed successfully!")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("Dibuat Oleh Kelompok 3 Rombel 2:")
    st.markdown("""
- Hizkia Natanael Richardo (4611422053)
- Nabil Mutawakkil Qisthi (4611422054)
- Fathimah Az-Zahra Sanjani (4611422057)
- Melinda Wijaya (4611422060)
""")

if __name__ == "__main__":
    main()