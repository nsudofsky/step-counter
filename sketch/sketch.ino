// Gait trainer - UNO Q MCU side.
// 1) Reads the MPU-6050 (I2C on the SDA/SCL header pins) 50 times a second and sends every
//    sample to Python as Bridge.notify("imu", ax, ay, az, gx, gy, gz) (raw int16 counts).
// 2) Draws what Python decided on the 13x8 LED matrix, animated. Python calls:
//      show(icon)   0 standing, 1 walking, 2 running, 3 biking, 4 "?", 5 check mark
//      number(n)    0-999 (step count, recording countdown)
//
// Wiring (MPU-6050 / GY-521 board): VCC -> 3.3V, GND -> GND, SDA -> SDA, SCL -> SCL.
#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <Wire.h>

const unsigned long SAMPLE_MS = 20;      // 50 Hz, must match RATE in gait.py

// ---------------------------------------------------------------- MPU-6050
uint8_t mpuAddr = 0x68;                  // 0x69 if the AD0 pin is pulled high
bool mpuOK = false;

bool mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(mpuAddr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

bool mpuBegin() {
  const uint8_t addrs[2] = {0x68, 0x69};
  for (int k = 0; k < 2; k++) {
    mpuAddr = addrs[k];
    if (mpuWrite(0x6B, 0x01)) {          // PWR_MGMT_1: wake up, clock from gyro X
      delay(50);
      mpuWrite(0x1A, 0x03);              // CONFIG: 44 Hz low-pass filter (smooths jitter, keeps steps)
      mpuWrite(0x19, 0x13);              // SMPLRT_DIV: 1 kHz / 20 = 50 Hz
      mpuWrite(0x1B, 0x18);              // GYRO_CONFIG: +-2000 deg/s  (16.4 counts per deg/s)
      mpuWrite(0x1C, 0x18);              // ACCEL_CONFIG: +-16 g       (2048 counts per g) - running ankles hit 6g+
      return true;
    }
  }
  return false;
}

bool mpuRead(int16_t v[6]) {
  Wire.beginTransmission(mpuAddr);
  Wire.write(0x3B);                      // ACCEL_XOUT_H, then accel, temp, gyro (14 bytes)
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(mpuAddr, (uint8_t)14) != 14) return false;
  uint8_t b[14];
  for (int i = 0; i < 14; i++) b[i] = Wire.read();
  for (int i = 0; i < 3; i++) v[i] = (int16_t)(b[2 * i] << 8 | b[2 * i + 1]);           // accel
  for (int i = 0; i < 3; i++) v[3 + i] = (int16_t)(b[8 + 2 * i] << 8 | b[9 + 2 * i]);   // gyro (skip temp)
  return true;
}

// ---------------------------------------------------------------- LED matrix
const int COLS = 13, ROWS = 8;
const bool MIRROR_X = true;              // same as the ping-pong scoreboard on this board
const bool MIRROR_Y = false;
Arduino_LED_Matrix matrix;
uint8_t frame[COLS * ROWS];

// Two animation frames per icon ('#' = on)
const char* ICONS[6][2][ROWS] = {
  { // 0 standing
    {"......#......", ".....###.....", "....#.#.#....", "......#......",
     "......#......", ".....#.#.....", ".....#.#.....", ".....#.#....."},
    {"......#......", ".....###.....", "....#.#.#....", "......#......",
     "......#......", ".....#.#.....", ".....#.#.....", ".....#.#....."}},
  { // 1 walking
    {"......#......", ".....###.....", "....#.#.#....", "......#......",
     "......#......", ".....#.#.....", "....#...#....", "...#.....#..."},
    {"......#......", ".....##......", "......##.....", "......#......",
     "......#......", "......#......", "......##.....", "......#......"}},
  { // 2 running (leaning forward, speed lines)
    {".......##....", ".......#.....", "##....###.#..", ".....#.#.#...",
     "##.....#.....", "......#.#....", "##...#...##..", "....#........"},
    {".......##....", ".......#.....", "......###....", "##...#.#.#...",
     ".......#.....", "##.....##....", "......#.#....", "##...#..#...."}},
  { // 3 biking (pedal goes round)
    {"..##......#..", "...#.....#...", "...#######...", ".###.#...###.",
     "#...##..#...#", "#.#.#.#.#.#.#", "#...#.#.#...#", ".###.....###."},
    {"..##......#..", "...#.....#...", "...#######...", ".###.#...###.",
     "#...###.#...#", "#...#.#.#...#", "#...#...#...#", ".###.....###."}},
  { // 4 not sure / no model yet
    {".....###.....", "....#...#....", "........#....", ".......#.....",
     "......#......", "......#......", ".............", "......#......"},
    {".....###.....", "....#...#....", "........#....", ".......#.....",
     "......#......", "......#......", ".............", "......#......"}},
  { // 5 check mark (recording done)
    {".............", "...........#.", "..........#..", ".........#...",
     ".#.....#.#...", "..#...#.#....", "...#.#.......", "....#........"},
    {".............", "...........#.", "..........#..", ".........#...",
     ".#.....#.#...", "..#...#.#....", "...#.#.......", "....#........"}},
};
const unsigned long FRAME_MS[6] = {1000, 400, 180, 250, 1000, 1000};

const uint8_t DIGITS[10][5] = {
  {7,5,5,5,7}, {2,6,2,2,7}, {7,1,7,4,7}, {7,1,7,1,7}, {5,5,7,1,1},
  {7,4,7,1,7}, {7,4,7,5,7}, {7,1,1,1,1}, {7,5,7,5,7}, {7,5,7,1,7},
};

volatile int icon = 4;                   // what loop() animates; -1 = showing a number
volatile int shownNumber = 0;
volatile bool dirty = true;

void setPixel(int col, int row) {
  if (col < 0 || col >= COLS || row < 0 || row >= ROWS) return;
  if (MIRROR_X) col = COLS - 1 - col;
  if (MIRROR_Y) row = ROWS - 1 - row;
  frame[row * COLS + col] = 255;
}

void drawIcon(int i, int f) {
  memset(frame, 0, sizeof(frame));
  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++)
      if (ICONS[i][f][r][c] == '#') setPixel(c, r);
  matrix.draw(frame);
}

void drawNumber(int n) {                 // 0-999: up to three 3x5 digits
  memset(frame, 0, sizeof(frame));
  n = constrain(n, 0, 999);
  int d[3], len = 0;
  do { d[len++] = n % 10; n /= 10; } while (n > 0);
  int width = len * 4 - 1;
  int col0 = (COLS - width) / 2;
  for (int k = 0; k < len; k++) {
    int digit = d[len - 1 - k];
    for (int r = 0; r < 5; r++)
      for (int c = 0; c < 3; c++)
        if (DIGITS[digit][r] & (4 >> c)) setPixel(col0 + 4 * k + c, 1 + r);
  }
  matrix.draw(frame);
}

void drawNoSensor() {                    // big X = no MPU-6050 answering, check the wires
  memset(frame, 0, sizeof(frame));
  for (int i = 0; i < ROWS; i++) { setPixel(2 + i, i); setPixel(9 - i, i); }
  matrix.draw(frame);
}

void show(int i) {
  if (i < 0 || i > 5) return;
  if (i != icon) { icon = i; dirty = true; }
}

void number(int n) {
  icon = -1;
  shownNumber = n;
  dirty = true;
}

// ---------------------------------------------------------------- main
void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(8);
  Wire.begin();
  Wire.setClock(400000);
  mpuOK = mpuBegin();

  Bridge.begin();
  Bridge.provide("show", show);
  Bridge.provide("number", number);
}

unsigned long nextSample = 0, nextFrame = 0, nextRetry = 0;
int zeroCount = 0;
int animFrame = 0;

void loop() {
  unsigned long now = millis();

  if (!mpuOK) {
    if (now >= nextRetry) {              // keep trying so you can fix the wiring live
      drawNoSensor();
      mpuOK = mpuBegin();
      nextRetry = now + 1000;
      dirty = true;
    }
    return;
  }

  if ((long)(now - nextSample) >= 0) {
    nextSample += SAMPLE_MS;
    if ((long)(now - nextSample) > 100) nextSample = now + SAMPLE_MS;   // fell way behind: resync
    int16_t v[6];
    if (mpuRead(v)) {
      // A real accelerometer always feels gravity, so all zeros = the MPU-6050 reset itself
      // (loose power wire). Wake it back up after half a second of zeros.
      bool zero = !(v[0] | v[1] | v[2] | v[3] | v[4] | v[5]);
      zeroCount = zero ? zeroCount + 1 : 0;
      if (zeroCount > 25) { mpuOK = false; zeroCount = 0; }
      Bridge.notify("imu", (int)v[0], (int)v[1], (int)v[2], (int)v[3], (int)v[4], (int)v[5]);
    } else {
      mpuOK = false;                     // unplugged? -> X on the matrix until it's back
    }
  }

  int i = icon;
  if (dirty || (i >= 0 && now >= nextFrame)) {
    if (i < 0) {
      drawNumber(shownNumber);
    } else {
      animFrame = dirty ? 0 : 1 - animFrame;
      drawIcon(i, animFrame);
      nextFrame = now + FRAME_MS[i];
    }
    dirty = false;
  }
}
