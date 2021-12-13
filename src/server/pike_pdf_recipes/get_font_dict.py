import pikepdf

if __name__ == '__main__':
    pdf = pikepdf.open('quick_start_data/pdf/K15-1004.pdf')
    resources = pdf.pages[0]["/Resources"]
    fonts = resources["/Font"]

