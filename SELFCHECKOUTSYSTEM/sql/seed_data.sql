-- Create the products table
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    price REAL NOT NULL
);

-- Insert product data
INSERT INTO products (name, price) VALUES ('BUT_CHI_DIXON', 1.50);
INSERT INTO products (name, price) VALUES ('BUT_HIGHLIGHT_MNG_TIM', 2.00);
INSERT INTO products (name, price) VALUES ('BUT_HIGHLIGHT_RETRO_COLOR', 2.50);
INSERT INTO products (name, price) VALUES ('BUT_LONG_SHARPIE_XANH', 3.00);
INSERT INTO products (name, price) VALUES ('BUT_NUOC_CS_8623', 1.80);
INSERT INTO products (name, price) VALUES ('BUT_XOA_NUOC', 1.20);
INSERT INTO products (name, price) VALUES ('HO_DOUBLE_8GM', 0.99);
INSERT INTO products (name, price) VALUES ('KEP_BUOM_19MM', 0.75);
INSERT INTO products (name, price) VALUES ('KEP_BUOM_25MM', 0.90);
INSERT INTO products (name, price) VALUES ('NGOI_CHI_MNG_0.5_100PCS', 4.00);
INSERT INTO products (name, price) VALUES ('SO_TAY_A6', 2.20);
INSERT INTO products (name, price) VALUES ('THUOC_CAMPUS_15CM', 0.80);
INSERT INTO products (name, price) VALUES ('THUOC_DO_DO', 1.10);
INSERT INTO products (name, price) VALUES ('THUOC_PARABOL', 1.30);
INSERT INTO products (name, price) VALUES ('XOA_KEO_CAPYBARA_9566', 1.60);