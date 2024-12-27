import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QMessageBox
)
from PyQt5.QtCore import Qt
import sys
from datetime import datetime

# Dữ liệu mẫu sản phẩm
product_prices = {"Apple": 10, "Banana": 5, "Cherry": 20}
class_counts = {name: 0 for name in product_prices.keys()}

# Tải mô hình YOLOv5 (hoặc YOLOv8)
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='your_model.pt')  # Đường dẫn tới file .pt của bạn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("../weights/best_v8.pt").to(device)

# Màn hình quét sản phẩm với YOLO
class ScanProductWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scan Product")
        self.setGeometry(100, 100, 400, 300)

        # Layout chính
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Hướng dẫn quét
        self.scan_instruction = QLabel("Click 'Scan Product' to scan a product.")
        layout.addWidget(self.scan_instruction)

        # Nút quét sản phẩm
        scan_button = QPushButton("Scan Product")
        scan_button.clicked.connect(self.scan_product)
        layout.addWidget(scan_button)

    def scan_product(self):
        # Mở camera
        cap = cv2.VideoCapture(0)  # Mở camera mặc định
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", "Could not access the camera.")
            return

        self.scan_instruction.setText("Scanning... Please scan the product.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Dự đoán bằng YOLOv5
            results = model(frame)  # Dự đoán trực tiếp từ mô hình PyTorch
            pred = results.pred[0]  # Dự đoán cho batch đầu tiên (chỉ 1 ảnh)

            # Khám phá các đối tượng nhận diện được
            for det in pred:  # Duyệt qua từng đối tượng trong kết quả
                xmin, ymin, xmax, ymax, confidence, class_id = det[:6].tolist()

                if confidence > 0.5:  # Nếu độ tin cậy cao hơn ngưỡng 0.5
                    label = model.names[int(class_id)]
                    class_counts[label] += 1  # Thêm sản phẩm vào giỏ hàng
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0),
                                2)

            # Hiển thị kết quả quét
            cv2.imshow("Scan Product", frame)

            # Nếu người dùng nhấn 'q' thì thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Màn hình chỉnh sửa hoá đơn
class EditInvoiceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edit Invoice")
        self.setGeometry(100, 100, 400, 400)

        # Layout chính
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Tạo bảng sản phẩm
        self.product_table = QTableWidget(len(product_prices), 3)
        self.product_table.setHorizontalHeaderLabels(["Product", "Price", "Quantity"])
        self.product_table.setEditTriggers(QTableWidget.NoEditTriggers)

        # Điền dữ liệu vào bảng
        for row, (product, price) in enumerate(product_prices.items()):
            self.product_table.setItem(row, 0, QTableWidgetItem(product))
            self.product_table.setItem(row, 1, QTableWidgetItem(str(price)))
            quantity_item = QTableWidgetItem(str(class_counts.get(product, 0)))
            quantity_item.setFlags(Qt.ItemIsEditable | Qt.ItemIsEnabled)
            self.product_table.setItem(row, 2, quantity_item)
        layout.addWidget(self.product_table)

        # Nút tính tổng và xuất hóa đơn
        self.total_label = QLabel("Total Price: 0")
        self.total_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.total_label)

        calculate_button = QPushButton("Calculate Total")
        calculate_button.clicked.connect(self.calculate_total)
        layout.addWidget(calculate_button)

        export_button = QPushButton("Export Bill")
        export_button.clicked.connect(self.export_bill)
        layout.addWidget(export_button)

    def calculate_total(self):
        total_price = 0
        for row in range(self.product_table.rowCount()):
            product = self.product_table.item(row, 0).text()
            quantity = int(self.product_table.item(row, 2).text())
            price = product_prices.get(product, 0)
            total_price += quantity * price
        self.total_label.setText(f"Total Price: {total_price}")

    def export_bill(self):
        total_price, transaction_file_path = export_bill(class_counts, product_prices)
        QMessageBox.information(self, "Bill Exported",
                                f"Bill exported successfully! Total: {total_price}.\nFile saved at {transaction_file_path}")


# Hàm xuất hóa đơn
def export_bill(class_counts, product_prices):
    transaction_id = f"TXN{datetime.now().strftime('%Y%m%d%H%M%S')}"
    transaction_file_path = "bill.txt"
    total_price = 0

    with open(transaction_file_path, "w", encoding="utf-8") as file:
        file.write(f"STATIONARY SHOP\nBill exported at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Transaction ID: {transaction_id}\n")
        file.write("--------------------------------------------------\n")
        for product, count in class_counts.items():
            if count > 0:
                price = product_prices.get(product, 0)
                total_price += count * price
                file.write(f"{product}:\t\t{count} \t{price}\n\n")
        file.write(f"Total price:\t\t\t\t\t\t{total_price}\n")
        file.write(f"Bill saved at:\t\t\t\t{transaction_file_path}\n")

    return total_price, transaction_file_path


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScanProductWindow()
    window.show()
    sys.exit(app.exec_())
