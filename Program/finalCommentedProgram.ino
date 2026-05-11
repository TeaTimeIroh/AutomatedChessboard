/******** Section: Library Includes
* Description
*   Pulls in all required third-party and system libraries:
*   ESP32 system logging, FastAccelStepper motor control, I2C bus
*   driver, 16x2 I2C LCD display, TensorFlow Lite Micro inference
*   runtime, and the compiled chess neural-network model binary.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
#include "esp_log.h"
#include <FastAccelStepper.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include "chess_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"


/******** Section: Stepper Motor Pin Definitions
* Description
*   Assigns the STEP and DIR output pins for both CoreXY stepper
*   drivers (DRV8825, 1/16 microstepping) connected to the ESP32-S3.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
#define DIR1_PIN  42
#define STEP1_PIN 41
#define DIR2_PIN  40
#define STEP2_PIN 39


/******** Section: Path-Clearance Constants
* Description
*   PATH_THRESHOLD_FRAC controls how close an occupied square's centre
*   must be to the electromagnet travel line before it is classified as
*   a blocker needing relocation (0.55 ≈ 20 mm at 37.5 mm/square).
*   MAX_BLOCKERS caps the size of the temporary blocker arrays used in
*   findPathBlockersRoute() and executeMove().
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
#define PATH_THRESHOLD_FRAC  0.55f
#define MAX_BLOCKERS          6


/******** Section: I/O Pin Assignments
* Description
*   Byte-constant GPIO numbers for the two homing limit switches,
*   the two player confirm/e-stop push buttons, and the electromagnet
*   relay output. All inputs use internal pull-ups.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
const byte LIMIT_SWITCH_Y_PIN = 4;
const byte LIMIT_SWITCH_X_PIN = 1;
const byte WHITE_BTN_PIN      = 13;
const byte BLACK_BTN_PIN      = 14;
const byte ELECTROMAGNET_PIN  = 2;


/******** Section: Physical Travel Limits
* Description
*   Maximum step counts for the X and Y CoreXY axes, calibrated to
*   the physical board dimensions. Re-verify with homeAndMeasure() if
*   drift exceeds 200 steps after mechanical changes.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
const int32_t MAX_X_STEPS = 21530;
const int32_t MAX_Y_STEPS = 22332;


/******** Section: Speed and Acceleration Settings
* Description
*   Stepper speed (steps/sec) and acceleration (steps/sec²) values for
*   three operating modes: slow calibration/homing speed, full-speed
*   piece movement, and the slow-down zone threshold near the walls.
*   Do not change without re-verifying mechanical limits.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
const uint32_t CALIBRATION_SPEED = 1600;
const uint32_t ACCELERATION      = 50000;
const int32_t  SLOWDOWN_ZONE     = 4000;
const int32_t  STEPS_PER_SQUARE  = 3000;
const uint32_t MOVE_SPEED        = 8000;
const uint32_t MOVE_ACCEL        = 40000;


/******** Section: Multiplexer Pin Definitions
* Description
*   GPIO assignments for the four CD74HC4067 16-channel analog
*   multiplexers used to scan all 64 reed switches. S0-S3 select
*   the active channel; EN0-EN3 enable individual multiplexers;
*   SIG is the shared digital read line.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
#define PIN_S0  16
#define PIN_S1  17
#define PIN_S2  18
#define PIN_S3   8
#define PIN_SIG  9
#define PIN_EN0  5
#define PIN_EN1  6
#define PIN_EN2  7
#define PIN_EN3 15

#define NUM_MUX      4
#define NUM_CHANNELS 16


/******** Section: LCD Configuration
* Description
*   I2C address, SDA/SCL GPIO pin assignments, and character-grid
*   dimensions for the 16x2 LCD used for game status display.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
#define LCD_I2C_ADDRESS  0x27
#define LCD_SDA_PIN      47
#define LCD_SCL_PIN      48
#define LCD_COLS         16
#define LCD_ROWS          2


/******** Section: Timing and Debounce Constants
* Description
*   Millisecond periods and tick counts governing the reed-switch scan
*   loop, LCD scroll animation, per-cell debounce (4 ticks x 50 ms =
*   200 ms), the piece-settling window, and the e-stop debounce count
*   (20 samples x 5 ms = 100 ms minimum sustained press).
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
#define FAST_SCAN_MS           50
#define LCD_SCROLL_INTERVAL_MS 300
#define DEBOUNCE_TICKS          4
#define SETTLE_WINDOW_MS      350
#define ESTOP_DEBOUNCE_COUNT   20


/******** Section: Neural Network Dimensions
* Description
*   Scratch-arena size allocated in PSRAM for TFLite Micro inference,
*   and the total number of move labels in the model output layer
*   (64 from-squares × 64 to-squares = 4096).
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
#define NN_ARENA_SIZE  (128 * 1024)
#define NN_N_MOVES      4096


/******** Section: Global Objects and State Variables
* Description
*   All module-level instances and flags shared across functions.
*   Includes the FastAccelStepper engine and stepper pointers,
*   MUX scan result and debounce arrays, LCD content and scroll state,
*   board occupancy and piece-model arrays, the 3-state move-detector
*   variables, legal-move and cycling buffers, en-passant and castling
*   rights, game colour and turn enumerations, and TFLite handles.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
static const char* TAG = "MAIN";

FastAccelStepperEngine engine = FastAccelStepperEngine();
FastAccelStepper*      stepper1 = nullptr;
FastAccelStepper*      stepper2 = nullptr;

bool          isStopped         = false;
bool          isElectromagnetOn = false;
unsigned long lastPrintTime     = 0;

const int EN_PINS[NUM_MUX] = { PIN_EN0, PIN_EN1, PIN_EN2, PIN_EN3 };
bool      results[NUM_MUX][NUM_CHANNELS];

LiquidCrystal_I2C Lcd(LCD_I2C_ADDRESS, LCD_COLS, LCD_ROWS);
String        lcdContent      = "";
int           lcdScrollOffset = 0;
bool          lcdNeedsScroll  = false;
unsigned long lastLcdScroll   = 0;
unsigned long lastFastScan    = 0;

bool    boardState[NUM_MUX][NUM_CHANNELS];
bool    rawPrev[NUM_MUX][NUM_CHANNELS];
uint8_t sameCount[NUM_MUX][NUM_CHANNELS];
bool    boardInitialized  = false;
bool    suppressDetection = false;

enum MoveDetectState { DETECT_IDLE, PIECE_IN_AIR, PIECE_SETTLING };
MoveDetectState detectState  = DETECT_IDLE;
String          liftedSquare = "";

unsigned long settleStart      = 0;
String        settleBestSquare = "";
uint8_t       settleBestCount  = 0;

#define MAX_LEGAL_MOVES        30
#define LEGAL_CORRECT_MAX_DIST  2
String legalDests[MAX_LEGAL_MOVES];
int    numLegalDests = 0;

int  epFile   = -1;
int  epRank   = -1;
bool castleWK = true;
bool castleWQ = true;
bool castleBK = true;
bool castleBQ = true;

String pendingMove       = "";
String lastDisplayedMove = "";

#define MAX_CYCLE_CANDIDATES 4
String        cycleCandidates[MAX_CYCLE_CANDIDATES];
int           numCycleCandidates = 0;
int           cycleIdx           = 0;
bool          cyclingSource      = false;
unsigned long lastCycleTime      = 0;

enum GameColor { COLOR_NONE, COLOR_WHITE, COLOR_BLACK };
enum TurnState { TURN_WHITE, TURN_BLACK };
GameColor playerColor = COLOR_NONE;
TurnState currentTurn = TURN_WHITE;

char pieceBoard[8][8];

const tflite::Model*      tfl_model   = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor*             nn_inp      = nullptr;
TfLiteTensor*             nn_out      = nullptr;
uint8_t*                  nn_arena    = nullptr;
uint8_t*                  model_buf   = nullptr;

bool computerMoveNeeded = false;
bool computerThinking   = false;


/******** Section: Function Prototypes
* Description
*   Forward declarations for all functions defined in this file,
*   allowing each function to be called before its definition without
*   relying on definition order.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void    homeAndMeasure(bool printMeasurement);
bool    checkEStop();
void    getLogicalPos(int32_t &logX, int32_t &logY);
bool    squareToMotorPos(char file, char rank, int32_t &m1, int32_t &m2);
void    moveToSquare(char file, char rank);
void    executeMove(char f1, char r1, char f2, char r2);
void    muxSelectChannel(uint8_t ch);
void    muxEnable(int index);
void    muxDisableAll();
void    muxScanAll();
String  muxToSquare(int m, int ch);
String  muxToSquareLower(int m, int ch);
String  muxBuildLcdString();
void    lcdWriteRow(int row, const String& text, int offset);
void    lcdUpdateRow1();
void    lcdUpdate();
void    printBoardState();
void    detectMove();
void    initPieceBoard();
void    selectPlayerColor();
void    passOrConfirm(bool isWhiteBtn);
void    startNewGame();
bool    lmgIsWhite(char p);
bool    lmgIsBlack(char p);
bool    lmgIsFriendly(char piece, int f, int r);
bool    lmgIsEnemy(char piece, int f, int r);
void    lmgAddDest(int f, int r);
void    lmgSlide(char piece, int f, int r, int df, int dr);
void    computeLegalDests(int fileIdx, int rankIdx);
String  nearestLegal(const String& candidate);
int     collectSourceCandidates(const String& dest, bool forWhite);
bool    lmgIsAttackedBy(int f, int r, bool byWhite, char board[8][8]);
bool    lmgFindKing(bool white, char board[8][8], int &kf, int &kr);
bool    lmgInCheckAfterMove(int f1, int r1, int f2, int r2, bool epCapture, bool white);
void    filterChecks(int srcF, int srcR);
void    resetBoardStateForLegalDests();
void    advanceCycling();
void    updatePendingCandidates();
static void  physicalPickAndPlace(char f1, char r1, char f2, char r2);
static bool  findParkingSquare(char origF, char origR, char pathF1, char pathR1, char pathF2, char pathR2, char resF[], char resR[], int nRes, char &outF, char &outR);
static int   findPathBlockersRoute(char f1, char r1, char f2, char r2, char bF[], char bR[]);
static int   computeMagnetRoute(char f1, char r1, char f2, char r2, char wpF[], char wpR[]);
static int   countNearSegment(char f1, char r1, char f2, char r2);
static bool  squareInPath(char pf, char pr, char f1, char r1, char f2, char r2);
static float distToSegSq(float px, float py, float ax, float ay, float bx, float by);
static void  squareToStepCoords(char file, char rank, float &tx, float &ty);
void    initNeuralNetwork();
void    encodeBoardForModel(int8_t* buf);
void    doComputerMove();


/******** function getLogicalPos
* Purpose
*   Computes the logical (X, Y) carriage position in steps from the
*   homed origin by decoding the CoreXY motor-coordinate system.
* Arguments
*   logX    Reference to int32_t that receives the logical X position
*           (const int32_t&, by-reference). Increases moving right.
*   logY    Reference to int32_t that receives the logical Y position
*           (const int32_t&, by-reference). Increases moving away from h1.
* Results
*   - Populates logX and logY via reference. No return value.
* Hardware
*   Reads stepper1 and stepper2 current step positions.
* Software
*   Depends on global stepper1 and stepper2 FastAccelStepper pointers.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void getLogicalPos(int32_t &logX, int32_t &logY) {
  int32_t p1 = stepper1->getCurrentPosition();
  int32_t p2 = stepper2->getCurrentPosition();
  logX = (p1 + p2) / 2;
  logY = (p1 - p2) / 2;
}


/******** function muxSelectChannel
* Purpose
*   Sets the four address lines (S0-S3) of the active CD74HC4067
*   multiplexer to select one of its 16 input channels.
* Arguments
*   ch    Channel index to select (uint8_t, 0-15, by-value).
* Results
*   - Drives PIN_S0 through PIN_S3 to the binary representation of ch.
*   - No return value.
* Hardware
*   PIN_S0 (GPIO 16), PIN_S1 (GPIO 17), PIN_S2 (GPIO 18), PIN_S3 (GPIO 8).
* Software
*   None.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void muxSelectChannel(uint8_t ch) {
  digitalWrite(PIN_S0, (ch >> 0) & 1);
  digitalWrite(PIN_S1, (ch >> 1) & 1);
  digitalWrite(PIN_S2, (ch >> 2) & 1);
  digitalWrite(PIN_S3, (ch >> 3) & 1);
}


/******** function muxEnable
* Purpose
*   Enables exactly one of the four CD74HC4067 multiplexers by pulling
*   its enable pin LOW while disabling all others (HIGH).
* Arguments
*   index    Zero-based index of the multiplexer to enable (int, by-value).
* Results
*   - Sets EN_PINS[index] LOW and all remaining EN_PINS HIGH.
*   - No return value.
* Hardware
*   PIN_EN0 (GPIO 5), PIN_EN1 (GPIO 6), PIN_EN2 (GPIO 7), PIN_EN3 (GPIO 15).
* Software
*   Reads global EN_PINS[] array and NUM_MUX constant.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void muxEnable(int index) {
  for (int i = 0; i < NUM_MUX; i++) {
    digitalWrite(EN_PINS[i], (i == index) ? LOW : HIGH);
  }
}


/******** function muxDisableAll
* Purpose
*   Disables all four CD74HC4067 multiplexers simultaneously by driving
*   every enable pin HIGH.
* Arguments
*   None.
* Results
*   - All EN_PINS driven HIGH. No return value.
* Hardware
*   PIN_EN0 through PIN_EN3.
* Software
*   Reads global EN_PINS[] array and NUM_MUX constant.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void muxDisableAll() {
  for (int i = 0; i < NUM_MUX; i++) {
    digitalWrite(EN_PINS[i], HIGH);
  }
}


/******** function muxScanAll
* Purpose
*   Performs a complete scan of all 64 reed switches by iterating over
*   all four multiplexers and all 16 channels each, recording each
*   digital reading into the global results[][] array.
* Arguments
*   None.
* Results
*   - Populates results[NUM_MUX][NUM_CHANNELS]: true = switch closed
*     (piece present), false = switch open. No return value.
* Hardware
*   PIN_SIG (GPIO 9) read as digital input with internal pull-up.
*   All four MUX enable and address lines driven as outputs.
* Software
*   Calls muxEnable(), muxSelectChannel(), muxDisableAll().
*   Writes global results[][].
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void muxScanAll() {
  for (int m = 0; m < NUM_MUX; m++) {
    muxEnable(m);
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
      muxSelectChannel(ch);
      delayMicroseconds(10);
      results[m][ch] = (digitalRead(PIN_SIG) == LOW);
    }
  }
  muxDisableAll();
}


/******** function muxToSquare
* Purpose
*   Converts a multiplexer index and channel number into an uppercase
*   two-character algebraic chess square name (e.g. "E4").
* Arguments
*   m     Multiplexer index (int, 0-3, by-value).
*   ch    Channel index within that multiplexer (int, 0-15, by-value).
* Results
*   - Returns a String of the form "<File><Rank>" in uppercase,
*     where File is A-H and Rank is 1-8.
* Hardware
*   None.
* Software
*   None.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
String muxToSquare(int m, int ch) {
  char file = 'H' - (ch % 8);
  int  rank = (3 - m) * 2 + 1 + (ch / 8);
  String s = "";
  s += file;
  s += rank;
  return s;
}


/******** function muxToSquareLower
* Purpose
*   Converts a multiplexer index and channel number into a lowercase
*   two-character algebraic chess square name (e.g. "e4"), suitable
*   for internal board-state comparisons.
* Arguments
*   m     Multiplexer index (int, 0-3, by-value).
*   ch    Channel index within that multiplexer (int, 0-15, by-value).
* Results
*   - Returns a String of the form "<file><rank>" in lowercase,
*     where file is a-h and rank is 1-8.
* Hardware
*   None.
* Software
*   Calls muxToSquare().
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
String muxToSquareLower(int m, int ch) {
  String s = muxToSquare(m, ch);
  s.toLowerCase();
  return s;
}


/******** function muxBuildLcdString
* Purpose
*   Builds a comma-separated list of all currently occupied squares
*   by reading results[][] and collecting every active channel. Used
*   to populate LCD row 0 with live board occupancy.
* Arguments
*   None.
* Results
*   - Returns a String of comma-separated uppercase square names
*     (e.g. "A1,E4,H8"), or the string "Empty board" if no switches
*     are active in results[][].
* Hardware
*   None.
* Software
*   Reads global results[][]. Calls muxToSquare().
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
String muxBuildLcdString() {
  String s    = "";
  bool   first = true;
  for (int m = 0; m < NUM_MUX; m++) {
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
      if (results[m][ch]) {
        if (!first) s += ',';
        s    += muxToSquare(m, ch);
        first = false;
      }
    }
  }
  return s.length() ? s : "Empty board";
}


/******** function lcdWriteRow
* Purpose
*   Writes exactly LCD_COLS characters to one row of the LCD starting
*   at a specified character offset within the source string. Any
*   position past the end of the string is padded with a space.
* Arguments
*   row      LCD row index to write to (int, 0 or 1, by-value).
*   text     Source string to display (const String&, by-reference).
*   offset   Starting character index within text (int, by-value).
* Results
*   - Overwrites all LCD_COLS characters on the specified row.
*   - No return value.
* Hardware
*   16x2 I2C LCD at address LCD_I2C_ADDRESS (SDA GPIO 47, SCL GPIO 48).
* Software
*   Depends on global Lcd (LiquidCrystal_I2C) object.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void lcdWriteRow(int row, const String& text, int offset) {
  Lcd.setCursor(0, row);
  for (int i = 0; i < LCD_COLS; i++) {
    int idx = offset + i;
    Lcd.print((idx < (int)text.length()) ? text[idx] : ' ');
  }
}


/******** function lcdUpdateRow1
* Purpose
*   Refreshes LCD row 1 with the current move status. Priority order:
*   (1) "?->?" while the computer is thinking, (2) "<src>->?" while
*   a piece is in the air or settling, (3) the pending move awaiting
*   confirmation, (4) the last confirmed move, (5) "------" if idle.
*   White's content is right-aligned; Black's is left-aligned.
* Arguments
*   None.
* Results
*   - Writes a 16-character string to LCD row 1. No return value.
* Hardware
*   16x2 I2C LCD.
* Software
*   Reads globals: computerThinking, currentTurn, detectState,
*   liftedSquare, pendingMove, lastDisplayedMove. Uses global Lcd.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void lcdUpdateRow1() {
  if (computerThinking) {
    String prefix  = (currentTurn == TURN_WHITE) ? "W>" : "b>";
    String content = prefix + "?->?";
    Lcd.setCursor(0, 1);
    if (currentTurn == TURN_WHITE) {
      while ((int)content.length() < LCD_COLS) content = ' ' + content;
      Lcd.print(content.substring(content.length() - LCD_COLS));
    } else {
      while ((int)content.length() < LCD_COLS) content += ' ';
      Lcd.print(content.substring(0, LCD_COLS));
    }
    return;
  }

  String action;
  if (detectState == PIECE_IN_AIR && liftedSquare.length()) {
    action = liftedSquare + "->?";
  } else if (detectState == PIECE_SETTLING && liftedSquare.length()) {
    action = liftedSquare + "->?";
  } else if (pendingMove.length()) {
    action = pendingMove;
  } else if (lastDisplayedMove.length()) {
    action = lastDisplayedMove;
  } else {
    action = "------";
  }

  String prefix  = (currentTurn == TURN_WHITE) ? "W>" : "b>";
  String content = prefix + action;

  Lcd.setCursor(0, 1);
  if (currentTurn == TURN_WHITE) {
    while ((int)content.length() < LCD_COLS) content = ' ' + content;
    Lcd.print(content.substring(content.length() - LCD_COLS));
  } else {
    while ((int)content.length() < LCD_COLS) content += ' ';
    Lcd.print(content.substring(0, LCD_COLS));
  }
}


/******** function lcdUpdate
* Purpose
*   Refreshes both rows of the LCD. Row 0 shows the occupied-squares
*   string, scrolled horizontally if it exceeds LCD_COLS characters.
*   Row 1 shows the current move status via lcdUpdateRow1().
* Arguments
*   None.
* Results
*   - Updates both LCD rows. No return value.
* Hardware
*   16x2 I2C LCD.
* Software
*   Reads globals: lcdNeedsScroll, lcdContent, lcdScrollOffset.
*   Calls lcdWriteRow() and lcdUpdateRow1().
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void lcdUpdate() {
  if (!lcdNeedsScroll) {
    lcdWriteRow(0, lcdContent, 0);
  } else {
    String padded = lcdContent + "   ";
    int    len    = padded.length();
    Lcd.setCursor(0, 0);
    for (int i = 0; i < LCD_COLS; i++) {
      Lcd.print(padded[(lcdScrollOffset + i) % len]);
    }
  }
  lcdUpdateRow1();
}

/******** function printBoardState
* Purpose
*   Prints an ASCII-art representation of the current pieceBoard[][]
*   to the Serial monitor, with rank 8 at the top and rank 1 at the
*   bottom, files a-h left to right. Uppercase = white piece,
*   lowercase = black piece, '.' = empty square.
* Arguments
*   None.
* Results
*   - Writes a formatted 10-line board diagram to Serial. No return value.
* Hardware
*   None.
* Software
*   Reads global pieceBoard[8][8].
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void printBoardState() {
  Serial.println("  ┌─────────────────┐");
  for (int r = 7; r >= 0; r--) {
    Serial.printf("  │ %d ", r + 1);
    for (int f = 0; f < 8; f++) {
      Serial.printf(" %c", pieceBoard[f][r]);
    }
    Serial.println("  │");
  }
  Serial.println("  └─────────────────┘");
  Serial.println("      a b c d e f g h");
}


/******** function initPieceBoard
* Purpose
*   Resets pieceBoard[][] to the standard chess starting position.
*   All 64 squares are first cleared to '.', then the two back ranks
*   and both pawn ranks are populated with the correct piece characters.
*   Uppercase = white, lowercase = black.
* Arguments
*   None.
* Results
*   - Overwrites global pieceBoard[8][8] with the opening position.
*   - No return value.
* Hardware
*   None.
* Software
*   Writes global pieceBoard[8][8].
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void initPieceBoard() {
  for (int f = 0; f < 8; f++) {
    for (int r = 0; r < 8; r++) {
      pieceBoard[f][r] = '.';
    }
  }
  const char wBack[] = "RNBQKBNR";
  const char bBack[] = "rnbqkbnr";
  for (int f = 0; f < 8; f++) {
    pieceBoard[f][0] = wBack[f];
    pieceBoard[f][1] = 'P';
    pieceBoard[f][6] = 'p';
    pieceBoard[f][7] = bBack[f];
  }
}


/******** function selectPlayerColor
* Purpose
*   Blocks until the player presses either the WHITE or BLACK button,
*   then records their chosen side in the global playerColor variable
*   and updates the LCD and Serial monitor accordingly.
* Arguments
*   None.
* Results
*   - Sets global playerColor to COLOR_WHITE or COLOR_BLACK.
*   - Updates the LCD with the chosen side message. No return value.
* Hardware
*   WHITE_BTN_PIN (GPIO 13), BLACK_BTN_PIN (GPIO 14) read as digital
*   inputs. 16x2 I2C LCD written for confirmation message.
* Software
*   Writes global playerColor. Depends on global Lcd object.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void selectPlayerColor() {
  Lcd.clear();
  Lcd.setCursor(0, 0);
  Lcd.print("Choose your side");
  Lcd.setCursor(0, 1);
  Lcd.print("BLACK      WHITE");
  Serial.println("Press BLACK (GPIO14) or WHITE (GPIO13) to choose colour.");
  while (true) {
    bool bB = (digitalRead(BLACK_BTN_PIN) == LOW);
    bool bW = (digitalRead(WHITE_BTN_PIN) == LOW);
    if (bB && bW) {
      delay(50);
      continue;
    }
    if (bB) {
      playerColor = COLOR_BLACK;
      Lcd.clear();
      Lcd.setCursor(0, 0);
      Lcd.print("You play BLACK  ");
      Lcd.setCursor(0, 1);
      Lcd.print("White to move   ");
      Serial.println("Player: BLACK. White moves first.");
      break;
    }
    if (bW) {
      playerColor = COLOR_WHITE;
      Lcd.clear();
      Lcd.setCursor(0, 0);
      Lcd.print("You play WHITE  ");
      Lcd.setCursor(0, 1);
      Lcd.print("White to move   ");
      Serial.println("Player: WHITE. White moves first.");
      break;
    }
    delay(50);
  }
  delay(400);
}


/******** function startNewGame
* Purpose
*   Performs a complete soft reset of all game state, re-homes the
*   carriage, re-initialises the LCD (recovering from any I2C
*   corruption caused by heavy motor or electromagnet activity),
*   takes a fresh reed-switch snapshot, and prompts the player to
*   choose a colour. If the computer plays White it sets the
*   computerMoveNeeded flag before returning.
*
*   isStopped is reset AFTER homing completes so that any EMI spike
*   during the homing sequence cannot leave the main loop frozen.
* Arguments
*   None.
* Results
*   - Resets all global game-state variables to their starting values.
*   - Homes the carriage and re-synchronises boardState[][].
*   - Calls selectPlayerColor() and lcdUpdate(). No return value.
* Hardware
*   Both stepper motors (homing), LCD (I2C re-init), RGB LED (GPIO 38),
*   all four multiplexers (initial scan).
* Software
*   Calls homeAndMeasure(), muxScanAll(), initPieceBoard(),
*   selectPlayerColor(), lcdUpdate(). Writes numerous globals.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void startNewGame() {
  Serial.println("\n=== NEW GAME ===");

  detectState        = DETECT_IDLE;
  pendingMove        = "";
  lastDisplayedMove  = "";
  liftedSquare       = "";
  settleBestSquare   = "";
  settleBestCount    = 0;
  numCycleCandidates = 0;
  cycleIdx           = 0;
  cyclingSource      = false;
  numLegalDests      = 0;
  suppressDetection  = false;
  epFile             = -1;
  epRank             = -1;
  castleWK           = true;
  castleWQ           = true;
  castleBK           = true;
  castleBQ           = true;
  currentTurn        = TURN_WHITE;
  playerColor        = COLOR_NONE;
  computerMoveNeeded = false;
  computerThinking   = false;
  initPieceBoard();

  Wire.begin(LCD_SDA_PIN, LCD_SCL_PIN);
  Lcd.init();
  Lcd.backlight();
  Lcd.clear();
  Lcd.setCursor(0, 0);
  Lcd.print(" NEW GAME       ");
  Lcd.setCursor(0, 1);
  Lcd.print(" Homing...      ");

  rgbLedWrite(38, 128, 128, 0);
  homeAndMeasure(false);

  isStopped = false;
  rgbLedWrite(38, 0, 0, 255);

  muxScanAll();
  memcpy(boardState, results, sizeof(boardState));
  memcpy(rawPrev,    results, sizeof(rawPrev));
  memset(sameCount,  0,       sizeof(sameCount));
  boardInitialized = true;
  lcdContent       = muxBuildLcdString();
  lcdNeedsScroll   = (lcdContent.length() > (size_t)LCD_COLS);
  printBoardState();
  selectPlayerColor();
  if (playerColor == COLOR_BLACK) computerMoveNeeded = true;
  lcdUpdate();
}


/******** function hasAnyLegalMove
* Purpose
*   Checks whether the specified side has at least one legal move
*   available in the current position. Used to detect checkmate and
*   stalemate after each confirmed move.
* Arguments
*   forWhite    true = check White's moves, false = check Black's
*               (bool, by-value).
* Results
*   - Returns true if at least one legal destination exists for any
*     piece of the requested colour; false if the side is stalemated
*     or checkmated.
* Hardware
*   None.
* Software
*   Calls computeLegalDests() for every piece of the given colour.
*   Reads global pieceBoard[8][8] and numLegalDests.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
bool hasAnyLegalMove(bool forWhite) {
  for (int f = 0; f < 8; f++) {
    for (int r = 0; r < 8; r++) {
      char p = pieceBoard[f][r];
      if (p == '.') continue;
      if (forWhite != lmgIsWhite(p)) continue;
      computeLegalDests(f, r);
      if (numLegalDests > 0) return true;
    }
  }
  return false;
}


/******** function showGameOver
* Purpose
*   Determines whether the side that has no legal moves is in check
*   (checkmate) or not (stalemate), then prints the result to Serial
*   and displays the appropriate game-over message on the LCD.
* Arguments
*   forWhite    true if White is the side with no legal moves
*               (bool, by-value).
* Results
*   - Writes a two-line game-over message to the LCD.
*   - Prints the result to Serial. No return value.
* Hardware
*   16x2 I2C LCD.
* Software
*   Calls lmgFindKing() and lmgIsAttackedBy(). Reads global pieceBoard.
*   Depends on global Lcd object.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void showGameOver(bool forWhite) {
  int  kf, kr;
  lmgFindKing(forWhite, pieceBoard, kf, kr);
  bool inCheck = lmgIsAttackedBy(kf, kr, !forWhite, pieceBoard);
  Lcd.clear();
  if (inCheck) {
    Serial.printf("%s is checkmated!\n", forWhite ? "White" : "Black");
    Lcd.setCursor(0, 0);
    Lcd.print(forWhite ? "  Black Wins!   " : "  White Wins!   ");
    Lcd.setCursor(0, 1);
    Lcd.print("   Checkmate    ");
  } else {
    Serial.println("Stalemate — Draw!");
    Lcd.setCursor(0, 0);
    Lcd.print("     Draw!      ");
    Lcd.setCursor(0, 1);
    Lcd.print("   Stalemate    ");
  }
}


/******** function passOrConfirm
* Purpose
*   Handles a button press from either player. If the pressing player
*   is NOT the current turn's player, a 1.5-second hold triggers a new
*   game; a short press clears any pending move so the human can retry.
*   If the pressing player IS the current turn's player, a pending move
*   is confirmed and committed to pieceBoard[][], special moves (en
*   passant, castling) are resolved, castling rights and the en-passant
*   target are updated, and the turn is flipped. After the flip,
*   checkmate and stalemate are tested; if the computer now needs to
*   move, computerMoveNeeded is set.
* Arguments
*   isWhiteBtn    true if the WHITE button was pressed, false for BLACK
*                 (bool, by-value).
* Results
*   - May commit a pending move to pieceBoard[][] and flip currentTurn.
*   - May set computerMoveNeeded, call startNewGame(), or call
*     showGameOver(). Updates LCD row 1. No return value.
* Hardware
*   WHITE_BTN_PIN (GPIO 13), BLACK_BTN_PIN (GPIO 14) polled inside the
*   hold-detection loop.
* Software
*   Calls lmgIsWhite(), muxScanAll(), printBoardState(), lcdUpdate(),
*   lcdUpdateRow1(), startNewGame(), hasAnyLegalMove(), showGameOver().
*   Reads and writes numerous game-state globals.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void passOrConfirm(bool isWhiteBtn) {
  bool isCurrentPlayer = (isWhiteBtn  && currentTurn == TURN_WHITE) ||
                         (!isWhiteBtn && currentTurn == TURN_BLACK);

  if (!isCurrentPlayer) {
    Lcd.clear();
    Lcd.setCursor(0, 0);
    Lcd.print("Hold 1.5s:      ");
    Lcd.setCursor(0, 1);
    Lcd.print("  new game      ");
    unsigned long t0 = millis();
    while (millis() - t0 < 1500) {
      bool held = isWhiteBtn ? (digitalRead(WHITE_BTN_PIN) == LOW)
                             : (digitalRead(BLACK_BTN_PIN) == LOW);
      if (!held) {
        if (pendingMove.length() > 0 || detectState != DETECT_IDLE) {
          Serial.println("Opponent btn: clearing pending move.");
          pendingMove        = "";
          detectState        = DETECT_IDLE;
          liftedSquare       = "";
          settleBestSquare   = "";
          settleBestCount    = 0;
          numCycleCandidates = 0;
          cycleIdx           = 0;
          cyclingSource      = false;
          muxScanAll();
          memcpy(boardState, results, sizeof(boardState));
          memcpy(rawPrev,    results, sizeof(rawPrev));
          memset(sameCount,  0,       sizeof(sameCount));
        }
        lcdUpdate();
        return;
      }
      delay(20);
    }
    startNewGame();
    return;
  }

  if (pendingMove.length() >= 6) {
    char f1  = pendingMove[0];
    char r1  = pendingMove[1];
    char f2  = pendingMove[4];
    char r2  = pendingMove[5];
    int  fx1 = f1 - 'a';
    int  ry1 = r1 - '1';
    int  fx2 = f2 - 'a';
    int  ry2 = r2 - '1';
    char moved    = pieceBoard[fx1][ry1];
    char captured = pieceBoard[fx2][ry2];

    bool wasEP = (toupper(moved) == 'P' && fx1 != fx2 &&
                  abs(ry2 - ry1) == 1 && captured == '.');
    if (wasEP) pieceBoard[fx2][ry1] = '.';

    bool white     = lmgIsWhite(moved);
    bool wasCastle = (toupper(moved) == 'K' && abs(fx2 - fx1) == 2);
    if (wasCastle) {
      int bki = white ? 0 : 7;
      if (fx2 == 6) {
        pieceBoard[5][bki] = pieceBoard[7][bki];
        pieceBoard[7][bki] = '.';
      } else if (fx2 == 2) {
        pieceBoard[3][bki] = pieceBoard[0][bki];
        pieceBoard[0][bki] = '.';
      }
    }
    pieceBoard[fx2][ry2] = moved;
    pieceBoard[fx1][ry1] = '.';

    if (toupper(moved) == 'P' && abs(ry2 - ry1) == 2) {
      epFile = fx1;
      epRank = ry2 + (white ? -1 : 1);
    } else {
      epFile = -1;
      epRank = -1;
    }
    if (moved == 'K') {
      castleWK = false;
      castleWQ = false;
    }
    if (moved == 'k') {
      castleBK = false;
      castleBQ = false;
    }
    if (moved == 'R' || captured == 'R') {
      if (fx1 == 7 || fx2 == 7) castleWK = false;
      if (fx1 == 0 || fx2 == 0) castleWQ = false;
    }
    if (moved == 'r' || captured == 'r') {
      if (fx1 == 7 || fx2 == 7) castleBK = false;
      if (fx1 == 0 || fx2 == 0) castleBQ = false;
    }

    Serial.printf("Confirmed: %s  piece=%c", pendingMove.c_str(), moved);
    if (captured != '.') Serial.printf("  captures=%c", captured);
    if (wasEP)           Serial.print("  [en passant]");
    if (wasCastle)       Serial.print("  [castling]");
    Serial.println();
    Serial.printf("  EP: %s  Castle: WK=%d WQ=%d BK=%d BQ=%d\n",
                  (epFile >= 0) ? String((char)('a' + epFile) +
                                  String(epRank + 1)).c_str() : "-",
                  castleWK, castleWQ, castleBK, castleBQ);
    printBoardState();

    lastDisplayedMove  = pendingMove;
    pendingMove        = "";
    detectState        = DETECT_IDLE;
    liftedSquare       = "";
    settleBestSquare   = "";
    settleBestCount    = 0;
    numCycleCandidates = 0;
    cycleIdx           = 0;
    cyclingSource      = false;

    muxScanAll();
    memcpy(boardState, results, sizeof(boardState));
    memcpy(rawPrev,    results, sizeof(rawPrev));
    memset(sameCount,  0,       sizeof(sameCount));
  } else {
    bool humanTurn = (playerColor == COLOR_WHITE && currentTurn == TURN_WHITE) ||
                     (playerColor == COLOR_BLACK && currentTurn == TURN_BLACK);
    if (humanTurn) {
      Serial.println("No pending move yet.");
      return;
    }
    Serial.println("Turn passed.");
  }

  currentTurn = (currentTurn == TURN_WHITE) ? TURN_BLACK : TURN_WHITE;
  Serial.printf("Turn: %s\n", (currentTurn == TURN_WHITE) ? "WHITE" : "BLACK");

  bool isNowComputerTurn = (playerColor == COLOR_WHITE && currentTurn == TURN_BLACK) ||
                           (playerColor == COLOR_BLACK && currentTurn == TURN_WHITE);
  if (isNowComputerTurn) computerMoveNeeded = true;

  bool nextIsWhite = (currentTurn == TURN_WHITE);
  if (!computerMoveNeeded && !hasAnyLegalMove(nextIsWhite)) {
    showGameOver(nextIsWhite);
    return;
  }
  lcdUpdateRow1();
}

/******** function lmgIsWhite
* Purpose
*   Predicate that returns true if the given piece character represents
*   a white piece (any uppercase ASCII letter A-Z).
* Arguments
*   p    Piece character to test (char, by-value).
* Results
*   - Returns true if p is uppercase (white piece); false otherwise.
* Hardware
*   None.
* Software
*   None.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
bool lmgIsWhite(char p) {
  return p >= 'A' && p <= 'Z';
}


/******** function lmgIsBlack
* Purpose
*   Predicate that returns true if the given piece character represents
*   a black piece (any lowercase ASCII letter a-z).
* Arguments
*   p    Piece character to test (char, by-value).
* Results
*   - Returns true if p is lowercase (black piece); false otherwise.
* Hardware
*   None.
* Software
*   None.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
bool lmgIsBlack(char p) {
  return p >= 'a' && p <= 'z';
}


/******** function lmgIsFriendly
* Purpose
*   Returns true if square (f, r) is occupied by a piece that belongs
*   to the same colour as the reference piece. Used during move
*   generation to block sliding or stepping onto own pieces.
* Arguments
*   piece    Reference piece whose colour is compared (char, by-value).
*   f        File index 0-7 of the target square (int, by-value).
*   r        Rank index 0-7 of the target square (int, by-value).
* Results
*   - Returns true if the target square is on the board, occupied, and
*     holds a piece of the same colour as piece; false otherwise.
* Hardware
*   None.
* Software
*   Reads global pieceBoard[8][8]. Calls lmgIsWhite() and lmgIsBlack().
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
bool lmgIsFriendly(char piece, int f, int r) {
  if (f < 0 || f >= 8 || r < 0 || r >= 8) return false;
  char t = pieceBoard[f][r];
  if (t == '.') return false;
  return lmgIsWhite(piece) ? lmgIsWhite(t) : lmgIsBlack(t);
}


/******** function lmgIsEnemy
* Purpose
*   Returns true if square (f, r) is occupied by a piece that belongs
*   to the opposite colour from the reference piece. Used to detect
*   capturable squares during move generation.
* Arguments
*   piece    Reference piece whose colour is compared (char, by-value).
*   f        File index 0-7 of the target square (int, by-value).
*   r        Rank index 0-7 of the target square (int, by-value).
* Results
*   - Returns true if the target square is on the board, occupied, and
*     holds a piece of the opposite colour to piece; false otherwise.
* Hardware
*   None.
* Software
*   Reads global pieceBoard[8][8]. Calls lmgIsWhite() and lmgIsBlack().
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
bool lmgIsEnemy(char piece, int f, int r) {
  if (f < 0 || f >= 8 || r < 0 || r >= 8) return false;
  char t = pieceBoard[f][r];
  if (t == '.') return false;
  return lmgIsWhite(piece) ? lmgIsBlack(t) : lmgIsWhite(t);
}


/******** function lmgAddDest
* Purpose
*   Appends a square to the global legalDests[] array if it is within
*   board bounds and the array has not reached MAX_LEGAL_MOVES entries.
*   The square name is built from the file and rank indices and stored
*   as a two-character String (e.g. "e4").
* Arguments
*   f    File index 0-7 to add (int, by-value).
*   r    Rank index 0-7 to add (int, by-value).
* Results
*   - Appends one entry to legalDests[] and increments numLegalDests.
*   - No-ops silently if out of bounds or array is full.
*   - No return value.
* Hardware
*   None.
* Software
*   Writes global legalDests[] and numLegalDests.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void lmgAddDest(int f, int r) {
  if (f < 0 || f >= 8 || r < 0 || r >= 8) return;
  if (numLegalDests >= MAX_LEGAL_MOVES) return;
  char s[3];
  s[0] = 'a' + f;
  s[1] = '1' + r;
  s[2] = 0;
  legalDests[numLegalDests++] = String(s);
}


/******** function lmgSlide
* Purpose
*   Generates all legal destination squares for a sliding piece
*   (bishop, rook, or queen) moving in one direction defined by the
*   (df, dr) step vector. Stops at the board edge, at a friendly piece,
*   or after capturing an enemy piece.
* Arguments
*   piece    The sliding piece character (char, by-value).
*   f        File index 0-7 of the piece's current square (int, by-value).
*   r        Rank index 0-7 of the piece's current square (int, by-value).
*   df       File step per iteration: -1, 0, or +1 (int, by-value).
*   dr       Rank step per iteration: -1, 0, or +1 (int, by-value).
* Results
*   - Appends reachable squares to legalDests[] via lmgAddDest().
*   - No return value.
* Hardware
*   None.
* Software
*   Calls lmgIsFriendly(), lmgIsEnemy(), lmgAddDest().
*   Reads global pieceBoard[8][8].
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void lmgSlide(char piece, int f, int r, int df, int dr) {
  for (int i = 1; i < 8; i++) {
    int nf = f + df * i;
    int nr = r + dr * i;
    if (nf < 0 || nf >= 8 || nr < 0 || nr >= 8) break;
    if (lmgIsFriendly(piece, nf, nr)) break;
    lmgAddDest(nf, nr);
    if (lmgIsEnemy(piece, nf, nr)) break;
  }
}


/******** function computeLegalDests
* Purpose
*   Populates legalDests[] with all pseudo-legal destination squares for
*   the piece at (fileIdx, rankIdx), then filters out moves that leave
*   the moving side's king in check via filterChecks(). Handles pawns
*   (single/double advance, diagonal capture), knights, bishops, rooks,
*   queens, kings, en-passant captures, and both castling directions.
*   Results are logged to Serial via ESP_LOGI.
* Arguments
*   fileIdx    File index 0-7 of the piece to generate moves for
*              (int, by-value).
*   rankIdx    Rank index 0-7 of the piece (int, by-value).
* Results
*   - Overwrites global legalDests[] and sets numLegalDests.
*   - No return value.
* Hardware
*   None.
* Software
*   Calls lmgSlide(), lmgAddDest(), lmgIsFriendly(), lmgIsEnemy(),
*   lmgIsAttackedBy(), filterChecks(). Reads globals pieceBoard[][],
*   epFile, epRank, castleWK, castleWQ, castleBK, castleBQ.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void computeLegalDests(int fileIdx, int rankIdx) {
  numLegalDests = 0;
  if (fileIdx < 0 || fileIdx >= 8 || rankIdx < 0 || rankIdx >= 8) return;
  char piece = pieceBoard[fileIdx][rankIdx];
  if (piece == '.') return;

  char pt    = toupper(piece);
  bool white = lmgIsWhite(piece);
  int  dir   = white ? 1 : -1;

  switch (pt) {
    case 'P': {
      int nr = rankIdx + dir;
      if (nr >= 0 && nr < 8 && pieceBoard[fileIdx][nr] == '.') {
        lmgAddDest(fileIdx, nr);
        int startRank = white ? 1 : 6;
        if (rankIdx == startRank && pieceBoard[fileIdx][rankIdx + 2 * dir] == '.') {
          lmgAddDest(fileIdx, rankIdx + 2 * dir);
        }
      }
      for (int df = -1; df <= 1; df += 2) {
        if (lmgIsEnemy(piece, fileIdx + df, rankIdx + dir))
          lmgAddDest(fileIdx + df, rankIdx + dir);
      }
      break;
    }
    case 'N': {
      const int mv[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
      for (int i = 0; i < 8; i++) {
        if (!lmgIsFriendly(piece, fileIdx + mv[i][0], rankIdx + mv[i][1]))
          lmgAddDest(fileIdx + mv[i][0], rankIdx + mv[i][1]);
      }
      break;
    }
    case 'B': {
      lmgSlide(piece, fileIdx, rankIdx,  1,  1);
      lmgSlide(piece, fileIdx, rankIdx,  1, -1);
      lmgSlide(piece, fileIdx, rankIdx, -1,  1);
      lmgSlide(piece, fileIdx, rankIdx, -1, -1);
      break;
    }
    case 'R': {
      lmgSlide(piece, fileIdx, rankIdx,  1,  0);
      lmgSlide(piece, fileIdx, rankIdx, -1,  0);
      lmgSlide(piece, fileIdx, rankIdx,  0,  1);
      lmgSlide(piece, fileIdx, rankIdx,  0, -1);
      break;
    }
    case 'Q': {
      lmgSlide(piece, fileIdx, rankIdx,  1,  1);
      lmgSlide(piece, fileIdx, rankIdx,  1, -1);
      lmgSlide(piece, fileIdx, rankIdx, -1,  1);
      lmgSlide(piece, fileIdx, rankIdx, -1, -1);
      lmgSlide(piece, fileIdx, rankIdx,  1,  0);
      lmgSlide(piece, fileIdx, rankIdx, -1,  0);
      lmgSlide(piece, fileIdx, rankIdx,  0,  1);
      lmgSlide(piece, fileIdx, rankIdx,  0, -1);
      break;
    }
    case 'K': {
      for (int df = -1; df <= 1; df++) {
        for (int dr = -1; dr <= 1; dr++) {
          if (df == 0 && dr == 0) continue;
          if (!lmgIsFriendly(piece, fileIdx + df, rankIdx + dr))
            lmgAddDest(fileIdx + df, rankIdx + dr);
        }
      }
      break;
    }
  }

  if (pt == 'P' && epFile >= 0) {
    int epCaptureRank = epRank + dir;
    if (abs(fileIdx - epFile) == 1 && rankIdx == epRank) {
      lmgAddDest(epFile, epCaptureRank);
    }
  }

  if (pt == 'K') {
    int backRank = white ? 0 : 7;
    if (rankIdx == backRank && fileIdx == 4) {
      bool inCheck = lmgIsAttackedBy(4, backRank, !white, pieceBoard);
      if (!inCheck) {
        bool ksr = white ? castleWK : castleBK;
        if (ksr &&
            pieceBoard[5][backRank] == '.' &&
            pieceBoard[6][backRank] == '.' &&
            pieceBoard[7][backRank] == (white ? 'R' : 'r') &&
            !lmgIsAttackedBy(5, backRank, !white, pieceBoard) &&
            !lmgIsAttackedBy(6, backRank, !white, pieceBoard)) {
          lmgAddDest(6, backRank);
        }
        bool qsr = white ? castleWQ : castleBQ;
        if (qsr &&
            pieceBoard[3][backRank] == '.' &&
            pieceBoard[2][backRank] == '.' &&
            pieceBoard[1][backRank] == '.' &&
            pieceBoard[0][backRank] == (white ? 'R' : 'r') &&
            !lmgIsAttackedBy(3, backRank, !white, pieceBoard) &&
            !lmgIsAttackedBy(2, backRank, !white, pieceBoard)) {
          lmgAddDest(2, backRank);
        }
      }
    }
  }

  filterChecks(fileIdx, rankIdx);

  String dbg = "Legal dests for ";
  dbg += (char)('a' + fileIdx);
  dbg += (char)('1' + rankIdx);
  dbg += " (";
  dbg += piece;
  dbg += "):";
  for (int i = 0; i < numLegalDests; i++) {
    dbg += ' ';
    dbg += legalDests[i];
  }
  ESP_LOGI(TAG, "%s", dbg.c_str());
}


/******** function nearestLegal
* Purpose
*   Checks whether a candidate destination square appears in the current
*   legalDests[] array. Returns the candidate unchanged if found, or an
*   empty String if it is not a legal move.
* Arguments
*   candidate    Two-character square name to look up (const String&,
*                by-reference).
* Results
*   - Returns candidate if it is found in legalDests[].
*   - Returns "" and logs a discard message if it is not found.
* Hardware
*   None.
* Software
*   Reads globals legalDests[] and numLegalDests. Logs via ESP_LOGI.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
String nearestLegal(const String& candidate) {
  for (int i = 0; i < numLegalDests; i++) {
    if (legalDests[i] == candidate) return candidate;
  }
  ESP_LOGI(TAG, "Rule filter: %s not legal, discarding", candidate.c_str());
  return "";
}


/******** function collectSourceCandidates
* Purpose
*   Finds all pieces of the current colour that can legally reach the
*   given destination square. If liftedSquare is among the valid sources
*   it is trusted directly and no cycling is needed. Otherwise all
*   candidates are loaded into cycleCandidates[] for the ambiguous-source
*   cycling UI. legalDests[] is left set for cycleCandidates[0] on exit.
* Arguments
*   dest        Two-character destination square (const String&, by-ref).
*   forWhite    true = search White's pieces, false = Black's (bool).
* Results
*   - Populates cycleCandidates[] and sets numCycleCandidates.
*   - Recomputes legalDests[] for cycleCandidates[0] before returning.
*   - Returns the number of valid source candidates found (int).
* Hardware
*   None.
* Software
*   Calls computeLegalDests(). Reads globals pieceBoard[][], liftedSquare,
*   legalDests[], numLegalDests. Writes cycleCandidates[], numCycleCandidates.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
int collectSourceCandidates(const String& dest, bool forWhite) {
  numCycleCandidates = 0;
  if (dest.length() < 2) return 0;
  int dfi = dest[0] - 'a';
  int dri = dest[1] - '1';
  if (dfi < 0 || dfi >= 8 || dri < 0 || dri >= 8) return 0;

  String buf[MAX_CYCLE_CANDIDATES];
  int    nBuf = 0;

  for (int fi = 0; fi < 8 && nBuf < MAX_CYCLE_CANDIDATES; fi++) {
    for (int ri = 0; ri < 8 && nBuf < MAX_CYCLE_CANDIDATES; ri++) {
      char p = pieceBoard[fi][ri];
      if (p == '.') continue;
      if (forWhite != lmgIsWhite(p)) continue;
      if (fi == dfi && ri == dri) continue;
      computeLegalDests(fi, ri);
      for (int i = 0; i < numLegalDests; i++) {
        if (legalDests[i] == dest) {
          String sq;
          sq += (char)('a' + fi);
          sq += (char)('1' + ri);
          buf[nBuf++] = sq;
          break;
        }
      }
    }
  }
  if (nBuf == 0) return 0;

  for (int i = 0; i < nBuf; i++) {
    if (buf[i] == liftedSquare) {
      numCycleCandidates = 1;
      cycleCandidates[0] = liftedSquare;
      computeLegalDests(liftedSquare[0] - 'a', liftedSquare[1] - '1');
      return 1;
    }
  }

  numCycleCandidates = nBuf;
  for (int i = 0; i < nBuf; i++) cycleCandidates[i] = buf[i];
  computeLegalDests(cycleCandidates[0][0] - 'a', cycleCandidates[0][1] - '1');
  return nBuf;
}


/******** function lmgIsAttackedBy
* Purpose
*   Returns true if square (f, r) is attacked by any piece of the
*   specified colour on the provided board snapshot. Checks pawn
*   diagonal attacks, knight jumps, king adjacency, bishop/queen
*   diagonals, and rook/queen orthogonals in turn.
* Arguments
*   f         File index 0-7 of the square to test (int, by-value).
*   r         Rank index 0-7 of the square to test (int, by-value).
*   byWhite   true = test White's attacks, false = Black's (bool).
*   board     8x8 board array to scan (char[8][8], by-pointer).
* Results
*   - Returns true if any attacking piece is found; false otherwise.
* Hardware
*   None.
* Software
*   Reads the supplied board[][] array only; does not touch globals.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
bool lmgIsAttackedBy(int f, int r, bool byWhite, char board[8][8]) {
  char wp = byWhite ? 'P' : 'p';
  char wn = byWhite ? 'N' : 'n';
  char wb = byWhite ? 'B' : 'b';
  char wr = byWhite ? 'R' : 'r';
  char wq = byWhite ? 'Q' : 'q';
  char wk = byWhite ? 'K' : 'k';
  int  pd = byWhite ? -1  :  1;

  for (int df = -1; df <= 1; df += 2) {
    int sf = f + df;
    int sr = r + pd;
    if (sf >= 0 && sf < 8 && sr >= 0 && sr < 8 && board[sf][sr] == wp) return true;
  }

  const int kn[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
  for (int i = 0; i < 8; i++) {
    int nf = f + kn[i][0];
    int nr = r + kn[i][1];
    if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8 && board[nf][nr] == wn) return true;
  }

  for (int df = -1; df <= 1; df++) {
    for (int dr = -1; dr <= 1; dr++) {
      if (!df && !dr) continue;
      int nf = f + df;
      int nr = r + dr;
      if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8 && board[nf][nr] == wk) return true;
    }
  }

  const int diag[4][2] = {{1,1},{1,-1},{-1,1},{-1,-1}};
  for (int d = 0; d < 4; d++) {
    for (int i = 1; i < 8; i++) {
      int  nf = f + diag[d][0] * i;
      int  nr = r + diag[d][1] * i;
      if (nf < 0 || nf >= 8 || nr < 0 || nr >= 8) break;
      char t  = board[nf][nr];
      if (t != '.') {
        if (t == wb || t == wq) return true;
        break;
      }
    }
  }

  const int straight[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
  for (int d = 0; d < 4; d++) {
    for (int i = 1; i < 8; i++) {
      int  nf = f + straight[d][0] * i;
      int  nr = r + straight[d][1] * i;
      if (nf < 0 || nf >= 8 || nr < 0 || nr >= 8) break;
      char t  = board[nf][nr];
      if (t != '.') {
        if (t == wr || t == wq) return true;
        break;
      }
    }
  }
  return false;
}


/******** function lmgFindKing
* Purpose
*   Searches the provided board snapshot for the king of the specified
*   colour and returns its coordinates via reference parameters.
* Arguments
*   white    true = find White's king ('K'), false = Black's ('k')
*            (bool, by-value).
*   board    8x8 board array to search (char[8][8], by-pointer).
*   kf       Reference that receives the king's file index 0-7 (int&).
*   kr       Reference that receives the king's rank index 0-7 (int&).
* Results
*   - Sets kf and kr to the king's location and returns true if found.
*   - Returns false if no king of the requested colour exists on the board
*     (should not occur in a valid game state).
* Hardware
*   None.
* Software
*   Reads the supplied board[][] array only.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
bool lmgFindKing(bool white, char board[8][8], int &kf, int &kr) {
  char k = white ? 'K' : 'k';
  for (int f = 0; f < 8; f++) {
    for (int r = 0; r < 8; r++) {
      if (board[f][r] == k) {
        kf = f;
        kr = r;
        return true;
      }
    }
  }
  return false;
}


/******** function lmgInCheckAfterMove
* Purpose
*   Applies a proposed move to a temporary board copy and tests whether
*   the moving side's king is in check on the resulting position. Handles
*   en-passant capture removal. Used by filterChecks() to detect pinned
*   pieces and king walk-into-check.
* Arguments
*   f1          Source file index 0-7 (int, by-value).
*   r1          Source rank index 0-7 (int, by-value).
*   f2          Destination file index 0-7 (int, by-value).
*   r2          Destination rank index 0-7 (int, by-value).
*   epCapture   true if this move is an en-passant capture (bool).
*   white       true if the moving piece is White (bool, by-value).
* Results
*   - Returns true if the moving side's king would be in check after
*     the move; false if the king is safe.
* Hardware
*   None.
* Software
*   Calls lmgFindKing() and lmgIsAttackedBy(). Reads globals
*   pieceBoard[][] and epFile.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
bool lmgInCheckAfterMove(int f1, int r1, int f2, int r2, bool epCapture, bool white) {
  char tmp[8][8];
  memcpy(tmp, pieceBoard, 64);
  tmp[f2][r2] = tmp[f1][r1];
  tmp[f1][r1] = '.';
  if (epCapture) tmp[epFile][r1] = '.';
  int kf, kr;
  if (!lmgFindKing(white, tmp, kf, kr)) return false;
  return lmgIsAttackedBy(kf, kr, !white, tmp);
}


/******** function filterChecks
* Purpose
*   Iterates over legalDests[] and removes any entry whose move would
*   leave the moving side's own king in check. This automatically
*   handles absolutely pinned pieces — every destination of a pinned
*   piece fails the check test and is removed. En-passant capture
*   removal is detected by checking whether a pawn moves diagonally
*   to an empty square matching epFile.
* Arguments
*   srcF    File index 0-7 of the piece being moved (int, by-value).
*   srcR    Rank index 0-7 of the piece being moved (int, by-value).
* Results
*   - Overwrites legalDests[] in place with only the legal entries and
*     updates numLegalDests. Logs the count of removed moves.
*   - No return value.
* Hardware
*   None.
* Software
*   Calls lmgInCheckAfterMove(). Reads globals pieceBoard[][], epFile,
*   legalDests[], numLegalDests.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void filterChecks(int srcF, int srcR) {
  bool white = lmgIsWhite(pieceBoard[srcF][srcR]);
  int  kept  = 0;
  for (int i = 0; i < numLegalDests; i++) {
    int  df = legalDests[i][0] - 'a';
    int  dr = legalDests[i][1] - '1';
    bool ep = false;
    if (toupper(pieceBoard[srcF][srcR]) == 'P' && df != srcF &&
        pieceBoard[df][dr] == '.' && epFile == df) {
      ep = true;
    }
    if (!lmgInCheckAfterMove(srcF, srcR, df, dr, ep, white)) {
      legalDests[kept++] = legalDests[i];
    }
  }
  int removed = numLegalDests - kept;
  numLegalDests = kept;
  if (removed > 0) {
    ESP_LOGI(TAG, "filterChecks removed %d illegal moves (pins/checks)", removed);
  }
}


/******** function resetBoardStateForLegalDests
* Purpose
*   Clears the boardState, rawPrev, and sameCount entries for every
*   square currently listed in legalDests[]. Called immediately after
*   a lift is detected so that any ghost activation already present on
*   a legal destination square is forgotten, allowing the debouncer to
*   treat the subsequent real placement as a fresh event.
* Arguments
*   None.
* Results
*   - Sets boardState[][], rawPrev[][], and sameCount[][] to 0/false
*     for every cell that maps to a legal destination square.
*   - No return value.
* Hardware
*   None.
* Software
*   Reads globals legalDests[], numLegalDests. Writes boardState[][],
*   rawPrev[][], sameCount[][].
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void resetBoardStateForLegalDests() {
  for (int i = 0; i < numLegalDests; i++) {
    int fi = legalDests[i][0] - 'a';
    int ri = legalDests[i][1] - '1';
    int m  = 3 - ri / 2;
    int ch = (ri % 2) * 8 + (7 - fi);
    if (m >= 0 && m < NUM_MUX && ch >= 0 && ch < NUM_CHANNELS) {
      boardState[m][ch] = false;
      rawPrev[m][ch]    = false;
      sameCount[m][ch]  = 0;
    }
  }
}


/******** function advanceCycling
* Purpose
*   Advances the ambiguous-move cycling index by one every 3 seconds
*   while more than one candidate exists. Updates pendingMove to show
*   the next candidate destination (or source) and refreshes LCD row 1.
*   No-ops if fewer than two candidates exist or the interval has not
*   elapsed.
* Arguments
*   None.
* Results
*   - Increments cycleIdx modulo numCycleCandidates.
*   - Rewrites pendingMove with the new candidate. Updates LCD row 1.
*   - No return value.
* Hardware
*   None (LCD refresh via lcdUpdateRow1()).
* Software
*   Reads globals numCycleCandidates, lastCycleTime, cyclingSource,
*   cycleCandidates[], pendingMove. Calls lcdUpdateRow1().
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void advanceCycling() {
  if (numCycleCandidates < 2) return;
  if (millis() - lastCycleTime < 3000) return;
  lastCycleTime = millis();
  cycleIdx      = (cycleIdx + 1) % numCycleCandidates;
  if (cyclingSource) {
    String dest = pendingMove.substring(4);
    pendingMove  = cycleCandidates[cycleIdx] + "->" + dest;
  } else {
    String src  = pendingMove.substring(0, 2);
    pendingMove = src + "->" + cycleCandidates[cycleIdx];
  }
  Serial.printf("  Cycling %s: %s\n",
                cyclingSource ? "src" : "dst", pendingMove.c_str());
  lcdUpdateRow1();
}


/******** function detectMove
* Purpose
*   Implements the 3-state reed-switch move detector. Per-cell debounce
*   is applied to results[][] on every call. State transitions:
*     DETECT_IDLE     — a debounced empty event on a non-empty model
*                       square records liftedSquare and computes legal
*                       destinations, entering PIECE_IN_AIR.
*     PIECE_IN_AIR    — a debounced occupied event starts the settling
*                       window and records the first placement candidate,
*                       entering PIECE_SETTLING.
*     PIECE_SETTLING  — additional occupied events update the best
*                       candidate (highest sameCount wins). When the
*                       SETTLE_WINDOW_MS window expires the best active
*                       legal destination is committed to pendingMove
*                       and the state returns to DETECT_IDLE.
*   Ghost reed-switch activations from adjacent squares fade first
*   (lower sameCount) while the true destination stays activated,
*   producing the highest count and winning the candidate selection.
*   Returns early without action if motors are running, detection is
*   suppressed, or a pending move already awaits confirmation.
* Arguments
*   None.
* Results
*   - May set liftedSquare, detectState, pendingMove, and cycleCandidates.
*   - May call resetBoardStateForLegalDests(), collectSourceCandidates(),
*     nearestLegal(), printBoardState(), lcdUpdateRow1().
*   - No return value.
* Hardware
*   None directly; reads state derived from muxScanAll() in results[][].
* Software
*   Reads globals results[][], boardState[][], rawPrev[][], sameCount[][],
*   boardInitialized, suppressDetection, stepper1, stepper2, pendingMove,
*   detectState, liftedSquare, settleStart, settleBestSquare,
*   settleBestCount, legalDests[], numLegalDests, currentTurn.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void detectMove() {
  if (!boardInitialized || suppressDetection) return;
  if (stepper1->isRunning() || stepper2->isRunning()) return;
  if (pendingMove.length() > 0) return;

  if (detectState == PIECE_SETTLING &&
      millis() - settleStart >= SETTLE_WINDOW_MS) {

    String found[MAX_LEGAL_MOVES];
    int    nFound = 0;
    for (int i = 0; i < numLegalDests; i++) {
      int fi = legalDests[i][0] - 'a';
      int ri = legalDests[i][1] - '1';
      int mm = 3 - ri / 2;
      int cc = (ri % 2) * 8 + (7 - fi);
      if (mm >= 0 && mm < NUM_MUX && cc >= 0 && cc < NUM_CHANNELS && results[mm][cc]) {
        if (legalDests[i] != liftedSquare) found[nFound++] = legalDests[i];
      }
    }

    String chosen      = "";
    cyclingSource      = false;
    numCycleCandidates = 0;

    if (nFound == 1) {
      chosen = found[0];
    } else if (nFound > 1) {
      numCycleCandidates = min(nFound, MAX_CYCLE_CANDIDATES);
      for (int i = 0; i < numCycleCandidates; i++) cycleCandidates[i] = found[i];
      cycleIdx      = 0;
      lastCycleTime = millis();
      chosen        = found[0];
      Serial.printf("  >> Ambiguous dest: %d:", nFound);
      for (int i = 0; i < nFound; i++) Serial.printf(" %s", found[i].c_str());
      Serial.println(" — cycling dests every 2s");
    } else {
      String activeFound = "";
      for (int i = 0; i < numLegalDests; i++) {
        int fi = legalDests[i][0] - 'a';
        int ri = legalDests[i][1] - '1';
        int mm = 3 - ri / 2;
        int cc = (ri % 2) * 8 + (7 - fi);
        if (mm >= 0 && mm < NUM_MUX && cc >= 0 && cc < NUM_CHANNELS &&
            results[mm][cc] && legalDests[i] != liftedSquare) {
          activeFound = legalDests[i];
          break;
        }
      }
      if (activeFound.length() > 0) {
        chosen = activeFound;
        ESP_LOGI(TAG, "Fallback: using active sensor %s over settleBest %s",
                 activeFound.c_str(), settleBestSquare.c_str());
      } else {
        chosen = nearestLegal(settleBestSquare);
      }
    }

    if (chosen.length() > 0) {
      bool forWhite = (currentTurn == TURN_WHITE);
      int  nSrc     = collectSourceCandidates(chosen, forWhite);
      if (nSrc == 0) {
        chosen = "";
      } else if (nSrc == 1) {
        liftedSquare = cycleCandidates[0];
      } else {
        cyclingSource = true;
        cycleIdx      = 0;
        lastCycleTime = millis();
        liftedSquare  = cycleCandidates[0];
        Serial.printf("  >> Ambiguous src: %d:", nSrc);
        for (int i = 0; i < nSrc; i++) Serial.printf(" %s", cycleCandidates[i].c_str());
        Serial.println(" — cycling sources every 2s");
      }
    }

    if (chosen.length() == 0 || chosen == liftedSquare) {
      ESP_LOGI(TAG, "Settled: no valid move, discarding");
      detectState        = DETECT_IDLE;
      liftedSquare       = "";
      settleBestSquare   = "";
      settleBestCount    = 0;
      numCycleCandidates = 0;
      cyclingSource      = false;
      return;
    }

    pendingMove  = liftedSquare + "->" + chosen;
    detectState  = DETECT_IDLE;
    liftedSquare = "";
    ESP_LOGI(TAG, "Pending: %s%s", pendingMove.c_str(),
             numCycleCandidates > 1
               ? (cyclingSource ? " [src-cyc]" : " [dst-cyc]")
               : "");
    Serial.printf("  >> Pending: %s  (press colour button to confirm)\n",
                  pendingMove.c_str());
    printBoardState();
    lcdUpdateRow1();
    return;
  }

  for (int m = 0; m < NUM_MUX; m++) {
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
      bool raw = results[m][ch];

      if (raw == rawPrev[m][ch]) {
        if (sameCount[m][ch] < 255) sameCount[m][ch]++;
      } else {
        sameCount[m][ch] = 1;
        rawPrev[m][ch]   = raw;
      }

      if (sameCount[m][ch] < DEBOUNCE_TICKS) continue;
      if (raw == boardState[m][ch])           continue;

      boardState[m][ch] = raw;
      String sq         = muxToSquareLower(m, ch);

      if (!raw) {
        if (detectState == DETECT_IDLE) {
          int lfi = sq[0] - 'a';
          int lri = sq[1] - '1';
          if (lfi < 0 || lfi >= 8 || lri < 0 || lri >= 8) continue;
          if (pieceBoard[lfi][lri] == '.') {
            ESP_LOGI(TAG, "Ghost lift ignored at %s (empty in model)", sq.c_str());
            continue;
          }
          liftedSquare      = sq;
          detectState       = PIECE_IN_AIR;
          settleBestSquare  = "";
          settleBestCount   = 0;
          computeLegalDests(sq[0] - 'a', sq[1] - '1');
          resetBoardStateForLegalDests();
          ESP_LOGI(TAG, "Lifted from %s (%d legal dests)", sq.c_str(), numLegalDests);
          lastDisplayedMove = "";
          lcdUpdateRow1();
        }
      } else if (detectState == PIECE_IN_AIR) {
        settleStart      = millis();
        detectState      = PIECE_SETTLING;
        settleBestSquare = sq;
        settleBestCount  = sameCount[m][ch];
        ESP_LOGI(TAG, "Settling started, candidate: %s (count=%d)",
                 sq.c_str(), sameCount[m][ch]);
        lcdUpdateRow1();
      } else if (detectState == PIECE_SETTLING) {
        if (sameCount[m][ch] > settleBestCount) {
          settleBestSquare = sq;
          settleBestCount  = sameCount[m][ch];
          ESP_LOGI(TAG, "Better candidate: %s (count=%d)",
                   sq.c_str(), sameCount[m][ch]);
        }
      }
    }
  }
}

/******** function initNeuralNetwork
* Purpose
*   Initialises the TensorFlow Lite Micro inference engine for the chess
*   move model. Copies the model binary from flash into PSRAM to eliminate
*   cache thrashing during inference (≈90 ms vs ≈825 ms in flash). Parses
*   the model, registers the required ops, allocates the scratch arena, and
*   constructs a static MicroInterpreter that outlives setup(). Logs tensor
*   shapes, quantisation parameters, and memory usage to Serial.
* Arguments
*   None.
* Results
*   - Allocates model_buf and nn_arena in PSRAM.
*   - Sets global pointers tfl_model, interpreter, nn_inp, nn_out.
*   - Returns early (globals left null) on any allocation or schema error.
*   - No return value.
* Hardware
*   Requires OPI PSRAM enabled in Tools menu (ESP32-S3 N8R8).
* Software
*   Reads global chess_model_tflite / chess_model_tflite_len (from header).
*   Writes globals model_buf, nn_arena, tfl_model, interpreter, nn_inp,
*   nn_out. Depends on TFLite Micro and NN_ARENA_SIZE / NN_N_MOVES defines.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void initNeuralNetwork() {
  Serial.println("\n=== Neural Network Init ===");

  if (!psramFound()) {
    Serial.println("ERROR: PSRAM not found — set Tools→PSRAM→OPI PSRAM");
  }
  Serial.printf("PSRAM free  : %.2f MB\n", ESP.getFreePsram() / 1e6f);
  Serial.printf("Model size  : %.2f MB (%u bytes)\n",
                chess_model_tflite_len / 1e6f, chess_model_tflite_len);

  Serial.print("Copying model flash→PSRAM... ");
  model_buf = (uint8_t*) ps_malloc(chess_model_tflite_len);
  if (!model_buf) {
    Serial.println("FAILED (ps_malloc)");
    return;
  }
  memcpy(model_buf, chess_model_tflite, chess_model_tflite_len);
  Serial.printf("OK  (PSRAM free: %.2f MB)\n", ESP.getFreePsram() / 1e6f);

  tfl_model = tflite::GetModel(model_buf);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Schema mismatch: model=%d expected=%d\n",
                  tfl_model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<7> resolver;
  resolver.AddConv2D();
  resolver.AddAdd();
  resolver.AddRelu();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddPad();
  resolver.AddTranspose();

  nn_arena = (uint8_t*) ps_malloc(NN_ARENA_SIZE);
  if (!nn_arena) {
    Serial.println("ERROR: ps_malloc for arena failed");
    return;
  }

  static tflite::MicroInterpreter static_interp(
      tfl_model, resolver, nn_arena, NN_ARENA_SIZE);
  interpreter = &static_interp;

  Serial.print("Allocating tensors... ");
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("FAILED — increase NN_ARENA_SIZE");
    return;
  }
  Serial.printf("OK  (used %u / %u bytes)\n",
                interpreter->arena_used_bytes(), NN_ARENA_SIZE);

  nn_inp = interpreter->input(0);
  nn_out = interpreter->output(0);
  Serial.printf("Input : [%d,%d,%d,%d] %s  scale=%.6f  zp=%d\n",
                nn_inp->dims->data[0], nn_inp->dims->data[1],
                nn_inp->dims->data[2], nn_inp->dims->data[3],
                TfLiteTypeGetName(nn_inp->type),
                nn_inp->params.scale, nn_inp->params.zero_point);
  Serial.printf("Output: [%d,%d] %s  scale=%.6f  zp=%d\n",
                nn_out->dims->data[0], nn_out->dims->data[1],
                TfLiteTypeGetName(nn_out->type),
                nn_out->params.scale, nn_out->params.zero_point);
  Serial.println("=== Neural Network Ready ===\n");
}


/******** function encodeBoardForModel
* Purpose
*   Serialises the current pieceBoard[][] into the 18-plane INT8 NHWC
*   tensor format expected by the chess model. Planes 0-5 carry the
*   moving side's pieces (P N B R Q K), planes 6-11 the opponent's.
*   Plane 12 is a constant side-to-move flag. Planes 13-16 encode the
*   four castling rights. Plane 17 marks the en-passant target square.
*   When Black is to move the board is flipped vertically so the model
*   always reasons from the perspective of the side to move.
*   Quantisation: float 0.0 maps to INT8 -128 (background);
*   float 1.0 maps to INT8 127 (piece or active flag).
* Arguments
*   buf    Pointer to an int8_t buffer of size 8*8*18 that receives
*          the encoded tensor (int8_t*, by-pointer).
* Results
*   - Fills buf[0 .. 8*8*18-1] with the quantised board encoding.
*   - No return value.
* Hardware
*   None.
* Software
*   Reads globals pieceBoard[][], currentTurn, castleWK, castleWQ,
*   castleBK, castleBQ, epFile, epRank.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void encodeBoardForModel(int8_t* buf) {
  memset(buf, (int8_t)(-128), 8 * 8 * 18);

  bool flip = (currentTurn == TURN_BLACK);

  for (int col = 0; col < 8; col++) {
    for (int row = 0; row < 8; row++) {
      char p = pieceBoard[col][row];
      if (p == '.') continue;
      auto charToPlane = [](char c) -> int {
        switch (c) {
          case 'P': return 0;
          case 'N': return 1;
          case 'B': return 2;
          case 'R': return 3;
          case 'Q': return 4;
          case 'K': return 5;
          case 'p': return 6;
          case 'n': return 7;
          case 'b': return 8;
          case 'r': return 9;
          case 'q': return 10;
          case 'k': return 11;
          default:  return -1;
        }
      };
      int plane = charToPlane(p);
      if (plane < 0) continue;
      if (flip) plane = (plane + 6) % 12;
      int drow = flip ? (7 - row) : row;
      buf[drow * 8 * 18 + col * 18 + plane] = 127;
    }
  }

  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 8; c++) {
      buf[r * 8 * 18 + c * 18 + 12] = 127;
    }
  }

  bool ourKS, ourQS, theirKS, theirQS;
  if (!flip) {
    ourKS   = castleWK;
    ourQS   = castleWQ;
    theirKS = castleBK;
    theirQS = castleBQ;
  } else {
    ourKS   = castleBK;
    ourQS   = castleBQ;
    theirKS = castleWK;
    theirQS = castleWQ;
  }
  auto fillPlane = [&](int plane) {
    for (int r = 0; r < 8; r++) {
      for (int c = 0; c < 8; c++) {
        buf[r * 8 * 18 + c * 18 + plane] = 127;
      }
    }
  };
  if (ourKS)   fillPlane(13);
  if (ourQS)   fillPlane(14);
  if (theirKS) fillPlane(15);
  if (theirQS) fillPlane(16);

  if (epFile >= 0 && epRank >= 0) {
    int drow = flip ? (7 - epRank) : epRank;
    buf[drow * 8 * 18 + epFile * 18 + 17] = 127;
  }
}


/******** function doComputerMove
* Purpose
*   Runs a complete computer turn: encodes the board, invokes the TFLite
*   model, then iterates output labels from highest to lowest INT8 score
*   until a fully legal move is found. Decodes the label into file/rank
*   characters, corrects Chess960 castling notation to standard king
*   destinations, verifies source piece colour and legality, then calls
*   executeMove() to physically perform the move. Displays "?->?" on the
*   LCD while the model runs and shows the chosen move for 600 ms before
*   the carriage starts moving. Auto-promotes any pawn reaching the back
*   rank to a queen in the board model.
* Arguments
*   None.
* Results
*   - Calls executeMove() with the best legal move found.
*   - Updates pieceBoard[][] for promotions.
*   - Calls showGameOver() if no legal move exists.
*   - Sets computerThinking, lastDisplayedMove, updates LCD. No return value.
* Hardware
*   CoreXY carriage and electromagnet (via executeMove()).
*   16x2 LCD (move preview and "thinking" display).
* Software
*   Calls encodeBoardForModel(), lcdUpdateRow1(), executeMove(),
*   computeLegalDests(), nearestLegal(), lmgIsWhite(), lmgIsBlack(),
*   showGameOver(), printBoardState(). Reads globals nn_inp, nn_out,
*   interpreter, currentTurn, pieceBoard[][], isStopped.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void doComputerMove() {
  Serial.println("\n=== COMPUTER THINKING ===");

  computerThinking = true;
  lcdUpdateRow1();

  encodeBoardForModel(nn_inp->data.int8);

  uint32_t t0 = millis();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: inference failed!");
    computerThinking = false;
    lcdUpdateRow1();
    return;
  }
  Serial.printf("Inference: %lu ms\n", millis() - t0);

  int8_t* raw  = nn_out->data.int8;
  bool    flip = (currentTurn == TURN_BLACK);

  static int8_t scores_copy[NN_N_MOVES];
  memcpy(scores_copy, raw, NN_N_MOVES);

  char bestF1 = 0;
  char bestR1 = 0;
  char bestF2 = 0;
  char bestR2 = 0;
  bool found  = false;

  for (int attempt = 0; attempt < NN_N_MOVES && !found; attempt++) {
    int best = 0;
    for (int i = 1; i < NN_N_MOVES; i++) {
      if (scores_copy[i] > scores_copy[best]) best = i;
    }
    if (scores_copy[best] == (int8_t)(-128)) break;

    int from_sq = best / 64;
    int to_sq   = best % 64;

    if (flip) {
      from_sq = (7 - from_sq / 8) * 8 + (from_sq % 8);
      to_sq   = (7 - to_sq   / 8) * 8 + (to_sq   % 8);
    }

    if      (from_sq == 4  && to_sq == 7)  to_sq = 6;
    else if (from_sq == 4  && to_sq == 0)  to_sq = 2;
    else if (from_sq == 60 && to_sq == 63) to_sq = 62;
    else if (from_sq == 60 && to_sq == 56) to_sq = 58;

    char f1 = 'a' + (char)(from_sq % 8);
    char r1 = '1' + (char)(from_sq / 8);
    char f2 = 'a' + (char)(to_sq   % 8);
    char r2 = '1' + (char)(to_sq   / 8);

    if (f1 < 'a' || f1 > 'h' || r1 < '1' || r1 > '8' ||
        f2 < 'a' || f2 > 'h' || r2 < '1' || r2 > '8') {
      scores_copy[best] = (int8_t)(-128);
      continue;
    }

    char piece      = pieceBoard[f1 - 'a'][r1 - '1'];
    bool rightColor = (currentTurn == TURN_WHITE)
                        ? lmgIsWhite(piece)
                        : lmgIsBlack(piece);
    if (piece == '.' || !rightColor) {
      scores_copy[best] = (int8_t)(-128);
      continue;
    }

    computeLegalDests(f1 - 'a', r1 - '1');
    String dest = String(f2) + String(r2);
    if (nearestLegal(dest).length() > 0) {
      bestF1 = f1;
      bestR1 = r1;
      bestF2 = f2;
      bestR2 = r2;
      found  = true;
    } else {
      scores_copy[best] = (int8_t)(-128);
    }
  }

  computerThinking = false;

  if (!found) {
    showGameOver(currentTurn == TURN_WHITE);
    return;
  }

  Serial.printf("Computer plays: %c%c->%c%c\n", bestF1, bestR1, bestF2, bestR2);

  lastDisplayedMove = String(bestF1) + String(bestR1) + "->" +
                      String(bestF2) + String(bestR2);
  lcdUpdateRow1();
  delay(600);

  char movingPiece = pieceBoard[bestF1 - 'a'][bestR1 - '1'];

  executeMove(bestF1, bestR1, bestF2, bestR2);

  if (!isStopped) {
    if ((movingPiece == 'P' && bestR2 == '8') ||
        (movingPiece == 'p' && bestR2 == '1')) {
      char queen = lmgIsWhite(movingPiece) ? 'Q' : 'q';
      pieceBoard[bestF2 - 'a'][bestR2 - '1'] = queen;
      Serial.printf("  [auto-promoted to %c]\n", queen);
      printBoardState();
    }
  }
}


/******** function setup
* Purpose
*   Arduino one-time initialisation. Configures all GPIO pins, brings up
*   both steppers, initialises the MUX address and enable lines, starts
*   I2C and the LCD, homes the carriage, takes the initial reed-switch
*   snapshot, sets the standard starting position in pieceBoard[][],
*   initialises the neural network, and prompts the player to choose a
*   colour. Sets computerMoveNeeded if the computer plays White.
* Arguments
*   None.
* Results
*   - All hardware peripherals initialised and ready.
*   - boardState[][] synchronised with the physical board.
*   - pieceBoard[][] set to the standard chess starting position.
*   - Neural network interpreter allocated and tensors ready.
*   - playerColor set; computerMoveNeeded set if computer plays White.
*   - No return value.
* Hardware
*   Both stepper motors and DRV8825 drivers, four CD74HC4067 MUXes,
*   16x2 I2C LCD (SDA GPIO 47, SCL GPIO 48), RGB LED (GPIO 38),
*   X and Y limit switches, WHITE/BLACK buttons, electromagnet output.
* Software
*   Calls homeAndMeasure(), muxScanAll(), initPieceBoard(),
*   initNeuralNetwork(), selectPlayerColor(), lcdUpdate().
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("=== SETUP START ===");

  pinMode(WHITE_BTN_PIN,      INPUT_PULLUP);
  pinMode(BLACK_BTN_PIN,      INPUT_PULLUP);
  pinMode(LIMIT_SWITCH_X_PIN, INPUT_PULLUP);
  pinMode(LIMIT_SWITCH_Y_PIN, INPUT_PULLUP);
  pinMode(ELECTROMAGNET_PIN,  OUTPUT);

  engine.init();
  stepper1 = engine.stepperConnectToPin(STEP1_PIN);
  stepper2 = engine.stepperConnectToPin(STEP2_PIN);
  if (!stepper1 || !stepper2) {
    Serial.println("FATAL: steppers failed!");
    delay(1000);
  }
  stepper1->setDirectionPin(DIR1_PIN);
  stepper2->setDirectionPin(DIR2_PIN);
  stepper1->setAcceleration(ACCELERATION);
  stepper2->setAcceleration(ACCELERATION);
  stepper1->setCurrentPosition(0);
  stepper2->setCurrentPosition(0);

  pinMode(PIN_S0, OUTPUT);
  pinMode(PIN_S1, OUTPUT);
  pinMode(PIN_S2, OUTPUT);
  pinMode(PIN_S3, OUTPUT);
  for (int i = 0; i < NUM_MUX; i++) {
    pinMode(EN_PINS[i], OUTPUT);
    digitalWrite(EN_PINS[i], HIGH);
  }
  pinMode(PIN_SIG, INPUT_PULLUP);
  muxSelectChannel(0);

  Wire.begin(LCD_SDA_PIN, LCD_SCL_PIN);
  Lcd.init();
  Lcd.backlight();
  Lcd.clear();
  Lcd.setCursor(0, 0);
  Lcd.print("  AutoChessboard");
  Lcd.setCursor(0, 1);
  Lcd.print("   Starting...  ");

  rgbLedWrite(38, 128, 128, 0);
  homeAndMeasure(false);
  rgbLedWrite(38, 0, 0, 255);

  muxScanAll();
  memcpy(boardState, results, sizeof(boardState));
  memcpy(rawPrev,    results, sizeof(rawPrev));
  memset(sameCount,  0,       sizeof(sameCount));
  boardInitialized = true;
  lcdContent       = muxBuildLcdString();
  lcdNeedsScroll   = (lcdContent.length() > (size_t)LCD_COLS);

  initPieceBoard();
  Serial.println("Piece board set to standard starting position.");
  printBoardState();

  initNeuralNetwork();

  selectPlayerColor();
  if (playerColor == COLOR_BLACK) computerMoveNeeded = true;
  lcdUpdate();

  ESP_LOGI(TAG, "Ready. MAX_X=%d MAX_Y=%d ZONE=%d",
           MAX_X_STEPS, MAX_Y_STEPS, (int)SLOWDOWN_ZONE);
  Serial.println("Commands: a1-h8 | e4-f3 | home");
}


/******** function loop
* Purpose
*   Arduino main loop, polled continuously. Responsibilities in order:
*   (1) Check for e-stop and return immediately if active.
*   (2) Trigger doComputerMove() when it is the computer's turn and the
*       motors are idle.
*   (3) Print carriage position to Serial every 250 ms via ESP_LOGI.
*   (4) Parse Serial commands: "pass", "home", "magnet_on/off", "board",
*       "f1r1-f2r2" (physical move), "f1r1->f2r2" (board-model update),
*       "fr" (jog to square).
*   (5) Every FAST_SCAN_MS: scan reed switches, run detectMove(),
*       updatePendingCandidates(), refresh LCD content and display.
*   (6) Advance the ambiguous-candidate cycling display.
*   (7) Handle WHITE/BLACK button presses (confirm or pass) when motors idle.
*   (8) Advance the LCD row-0 scroll animation.
* Arguments
*   None (Arduino framework calls this function repeatedly).
* Results
*   - Drives all runtime game logic, sensor polling, and UI updates.
*   - No return value.
* Hardware
*   All hardware peripherals (steppers, MUXes, LCD, buttons, electromagnet).
* Software
*   Calls checkEStop(), doComputerMove(), getLogicalPos(), muxScanAll(),
*   detectMove(), updatePendingCandidates(), muxBuildLcdString(),
*   lcdUpdate(), advanceCycling(), passOrConfirm(), homeAndMeasure(),
*   executeMove(), moveToSquare(), printBoardState(), lcdUpdateRow1().
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void loop() {
  if (checkEStop()) return;

  if (computerMoveNeeded && !isStopped &&
      !stepper1->isRunning() && !stepper2->isRunning()) {
    computerMoveNeeded = false;
    doComputerMove();
  }

  if (millis() - lastPrintTime > 250) {
    int32_t logX, logY;
    getLogicalPos(logX, logY);
    int32_t absLogX = abs(logX);
    ESP_LOGI(TAG, "L:%ld R:%ld B:%ld T:%ld",
             (long)absLogX,            (long)(MAX_X_STEPS - absLogX),
             (long)logY,               (long)(MAX_Y_STEPS - logY));
    lastPrintTime = millis();
  }

  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    cmd.toLowerCase();

    if (cmd == "pass") {
      currentTurn  = (currentTurn == TURN_WHITE) ? TURN_BLACK : TURN_WHITE;
      pendingMove  = "";
      detectState  = DETECT_IDLE;
      liftedSquare = "";
      Serial.printf("Turn forced to: %s\n",
                    (currentTurn == TURN_WHITE) ? "WHITE" : "BLACK");
      lcdUpdateRow1();

    } else if (cmd == "home") {
      rgbLedWrite(38, 128, 128, 0);
      homeAndMeasure(true);
      rgbLedWrite(38, 0, 0, 255);

    } else if (cmd == "magnet_on") {
      digitalWrite(ELECTROMAGNET_PIN, HIGH);
      Serial.println("magnet on");

    } else if (cmd == "magnet_off") {
      digitalWrite(ELECTROMAGNET_PIN, LOW);
      Serial.println("magnet off");

    } else if (cmd == "board") {
      printBoardState();

    } else if (cmd.length() == 5 && cmd[2] == '-') {
      char f1 = cmd[0];
      char r1 = cmd[1];
      char f2 = cmd[3];
      char r2 = cmd[4];
      if (f1 >= 'a' && f1 <= 'h' && r1 >= '1' && r1 <= '8' &&
          f2 >= 'a' && f2 <= 'h' && r2 >= '1' && r2 <= '8') {
        executeMove(f1, r1, f2, r2);
      } else {
        Serial.printf("Invalid move '%s'. Format: e4-f3\n", cmd.c_str());
      }

    } else if (cmd.length() == 6 && cmd[2] == '-' && cmd[3] == '>') {
      char f1 = cmd[0];
      char r1 = cmd[1];
      char f2 = cmd[4];
      char r2 = cmd[5];
      if (f1 >= 'a' && f1 <= 'h' && r1 >= '1' && r1 <= '8' &&
          f2 >= 'a' && f2 <= 'h' && r2 >= '1' && r2 <= '8') {
        int  fx1      = f1 - 'a';
        int  ry1      = r1 - '1';
        int  fx2      = f2 - 'a';
        int  ry2      = r2 - '1';
        char moved    = pieceBoard[fx1][ry1];
        char captured = pieceBoard[fx2][ry2];
        pieceBoard[fx2][ry2] = moved;
        pieceBoard[fx1][ry1] = '.';
        if (toupper(moved) == 'P' && abs(ry2 - ry1) == 2) {
          epFile = fx1;
          epRank = ry2 + (lmgIsWhite(moved) ? -1 : 1);
        } else {
          epFile = -1;
          epRank = -1;
        }
        if (moved == 'K') {
          castleWK = false;
          castleWQ = false;
        }
        if (moved == 'k') {
          castleBK = false;
          castleBQ = false;
        }
        if (moved == 'R' || captured == 'R') {
          if (fx1 == 7 || fx2 == 7) castleWK = false;
          if (fx1 == 0 || fx2 == 0) castleWQ = false;
        }
        if (moved == 'r' || captured == 'r') {
          if (fx1 == 7 || fx2 == 7) castleBK = false;
          if (fx1 == 0 || fx2 == 0) castleBQ = false;
        }
        lastDisplayedMove  = String(f1) + String(r1) + "->" +
                             String(f2) + String(r2);
        pendingMove        = "";
        detectState        = DETECT_IDLE;
        liftedSquare       = "";
        settleBestSquare   = "";
        settleBestCount    = 0;
        numCycleCandidates = 0;
        cyclingSource      = false;
        currentTurn        = (currentTurn == TURN_WHITE) ? TURN_BLACK : TURN_WHITE;
        bool isNowComputerTurn =
            (playerColor == COLOR_WHITE && currentTurn == TURN_BLACK) ||
            (playerColor == COLOR_BLACK && currentTurn == TURN_WHITE);
        if (isNowComputerTurn) computerMoveNeeded = true;
        Serial.printf("Manual move: %c%c->%c%c  piece=%c\n",
                      f1, r1, f2, r2, moved);
        printBoardState();
        lcdUpdateRow1();
      } else {
        Serial.printf("Invalid move '%s'. Format: e2->e4\n", cmd.c_str());
      }

    } else if (cmd.length() == 2) {
      char file = cmd[0];
      char rank = cmd[1];
      if (file >= 'a' && file <= 'h' && rank >= '1' && rank <= '8') {
        moveToSquare(file, rank);
      } else {
        Serial.printf("Unknown '%s'.\n", cmd.c_str());
      }

    } else {
      Serial.printf("Unknown '%s'. Commands: a1-h8 | e4-f3 | home | board\n",
                    cmd.c_str());
    }
  }

  unsigned long now = millis();
  if (now - lastFastScan >= FAST_SCAN_MS) {
    lastFastScan = now;
    muxScanAll();
    detectMove();
    updatePendingCandidates();
    String nc = muxBuildLcdString();
    if (nc != lcdContent) {
      lcdContent      = nc;
      lcdScrollOffset = 0;
      lcdNeedsScroll  = (lcdContent.length() > (size_t)LCD_COLS);
    }
    lcdUpdate();
  }

  advanceCycling();

  if (!stepper1->isRunning() && !stepper2->isRunning()) {
    bool bW = (digitalRead(WHITE_BTN_PIN) == LOW);
    bool bB = (digitalRead(BLACK_BTN_PIN) == LOW);
    if (!(bW && bB)) {
      if (bW) {
        passOrConfirm(true);
        delay(300);
      }
      if (bB) {
        passOrConfirm(false);
        delay(300);
      }
    }
  }

  if (lcdNeedsScroll && (now - lastLcdScroll >= LCD_SCROLL_INTERVAL_MS)) {
    lastLcdScroll = now;
    lcdScrollOffset++;
    lcdUpdate();
  }
}

/******** function checkEStop
* Purpose
*   Polls both player buttons and toggles the global isStopped flag when
*   both are held simultaneously for at least ESTOP_DEBOUNCE_COUNT
*   consecutive 5 ms samples (≈100 ms). The debounce count requirement
*   prevents EMI spikes induced by stepper phase switching or electromagnet
*   transients from triggering a false e-stop. A 400 ms block delay after
*   the toggle prevents rapid re-triggering from a sustained hold. When
*   isStopped is true both steppers are force-stopped on every call.
* Arguments
*   None.
* Results
*   - May toggle global isStopped and call stepper forceStop().
*   - Returns true if isStopped is currently active; false otherwise.
* Hardware
*   WHITE_BTN_PIN (GPIO 13), BLACK_BTN_PIN (GPIO 14), RGB LED (GPIO 38),
*   stepper1 and stepper2 via FastAccelStepper.
* Software
*   Reads globals isStopped, stepper1, stepper2. Uses static eStopCount.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
bool checkEStop() {
  static uint8_t eStopCount = 0;

  bool bW = (digitalRead(WHITE_BTN_PIN) == LOW);
  bool bB = (digitalRead(BLACK_BTN_PIN) == LOW);

  if (bW && bB) {
    if (++eStopCount >= ESTOP_DEBOUNCE_COUNT) {
      eStopCount = 0;
      isStopped  = !isStopped;
      delay(400);
      if (isStopped) {
        rgbLedWrite(38, 255, 0, 0);
        Serial.println("\n!!! EMERGENCY STOP — press both buttons again to resume !!!");
        stepper1->forceStop();
        stepper2->forceStop();
      } else {
        rgbLedWrite(38, 0, 0, 255);
        Serial.println("\n>>> SYSTEM READY");
      }
    }
  } else {
    eStopCount = 0;
  }

  if (isStopped) {
    stepper1->forceStop();
    stepper2->forceStop();
    return true;
  }
  return false;
}


/******** function squareToMotorPos
* Purpose
*   Converts an algebraic chess square into the two stepper target
*   positions required by the CoreXY kinematics. The result can be
*   passed directly to stepper moveTo() calls.
* Arguments
*   file    File character 'a'-'h' (char, by-value).
*   rank    Rank character '1'-'8' (char, by-value).
*   m1      Reference that receives stepper1 target position (int32_t&).
*   m2      Reference that receives stepper2 target position (int32_t&).
* Results
*   - Sets m1 and m2 via reference and returns true on success.
*   - Returns false without modifying m1/m2 if file or rank is out of range.
* Hardware
*   None.
* Software
*   Reads global STEPS_PER_SQUARE constant.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
bool squareToMotorPos(char file, char rank, int32_t &m1, int32_t &m2) {
  if (file < 'a' || file > 'h' || rank < '1' || rank > '8') return false;
  int32_t tx = (int32_t)(rank - '1')       * STEPS_PER_SQUARE;
  int32_t ty = (int32_t)(7 - (file - 'a')) * STEPS_PER_SQUARE;
  m1 = ty - tx;
  m2 = -(tx + ty);
  return true;
}


/******** function moveToSquare
* Purpose
*   Moves the CoreXY carriage to the centre of the specified chess square
*   using trapezoidal velocity profiling. Applies a reduced speed and
*   acceleration (CALIBRATION_SPEED / ACCELERATION) when the carriage is
*   within SLOWDOWN_ZONE steps of any wall, switching dynamically mid-move
*   if the slow zone boundary is crossed. Both steppers receive
*   proportionally scaled speeds so the gantry tracks a straight line.
*   The e-stop is polled every 5 ms inside the motion loop; if triggered
*   both motors are force-stopped and the function returns early.
* Arguments
*   file    Destination file character 'a'-'h' (char, by-value).
*   rank    Destination rank character '1'-'8' (char, by-value).
* Results
*   - Blocks until the carriage arrives or isStopped becomes true.
*   - Restores ACCELERATION on both steppers before returning.
*   - No return value.
* Hardware
*   stepper1 and stepper2 via FastAccelStepper.
* Software
*   Calls getLogicalPos(), checkEStop(). Reads globals MOVE_SPEED,
*   MOVE_ACCEL, CALIBRATION_SPEED, ACCELERATION, SLOWDOWN_ZONE,
*   MAX_X_STEPS, MAX_Y_STEPS, STEPS_PER_SQUARE, stepper1, stepper2.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void moveToSquare(char file, char rank) {
  if (file < 'a' || file > 'h' || rank < '1' || rank > '8') {
    Serial.printf("Invalid square: %c%c\n", file, rank);
    return;
  }
  int32_t logX, logY;
  getLogicalPos(logX, logY);
  float x0 = (float)abs(logX);
  float y0 = (float)logY;
  float x1 = (float)(rank - '1') * STEPS_PER_SQUARE;
  float y1 = (float)(7 - (file - 'a')) * STEPS_PER_SQUARE;
  float dx  = x1 - x0;
  float dy  = y1 - y0;
  if (fabsf(dx) < 0.5f && fabsf(dy) < 0.5f) {
    ESP_LOGI(TAG, "Already at %c%c", (int)file, (int)rank);
    return;
  }

  int32_t  m1T     = (int32_t)roundf(y1) - (int32_t)roundf(x1);
  int32_t  m2T     = -((int32_t)roundf(x1) + (int32_t)roundf(y1));
  int32_t  d1      = abs(m1T - stepper1->getCurrentPosition());
  int32_t  d2      = abs(m2T - stepper2->getCurrentPosition());
  int32_t  dom     = max(d1, d2);
  if (dom < 10) {
    ESP_LOGI(TAG, "Already at %c%c", (int)file, (int)rank);
    return;
  }

  float    r1      = (float)d1 / (float)dom;
  float    r2      = (float)d2 / (float)dom;
  uint32_t distCap = (uint32_t)sqrtf((float)MOVE_ACCEL * (float)dom);
  uint32_t safeSpd = min(MOVE_SPEED, distCap);

  auto inSlowZone = [&](float wx, float wy) -> bool {
    if (wx < SLOWDOWN_ZONE                     && dx < 0) return true;
    if (wx > (MAX_X_STEPS - SLOWDOWN_ZONE)     && dx > 0) return true;
    if (wy < SLOWDOWN_ZONE                     && dy < 0) return true;
    if (wy > (MAX_Y_STEPS - SLOWDOWN_ZONE)     && dy > 0) return true;
    return false;
  };

  auto applySpd = [&](uint32_t spd, uint32_t acc) {
    uint32_t s1 = (d1 > 0) ? max((uint32_t)(spd * r1), (uint32_t)200)  : 200;
    uint32_t s2 = (d2 > 0) ? max((uint32_t)(spd * r2), (uint32_t)200)  : 200;
    uint32_t a1 = (d1 > 0) ? max((uint32_t)(acc * r1), (uint32_t)1000) : 1000;
    uint32_t a2 = (d2 > 0) ? max((uint32_t)(acc * r2), (uint32_t)1000) : 1000;
    stepper1->setAcceleration(a1);
    stepper2->setAcceleration(a2);
    stepper1->setSpeedInHz(s1);
    stepper2->setSpeedInHz(s2);
    stepper1->applySpeedAcceleration();
    stepper2->applySpeedAcceleration();
  };

  bool curSlow = inSlowZone(x0, y0);
  applySpd(curSlow ? CALIBRATION_SPEED : safeSpd,
           curSlow ? ACCELERATION      : MOVE_ACCEL);
  stepper1->moveTo(m1T);
  stepper2->moveTo(m2T);
  ESP_LOGI(TAG, ">> %c%c safeSpd=%lu slow=%d",
           (int)file, (int)rank, (unsigned long)safeSpd, (int)curSlow);

  while (stepper1->isRunning() || stepper2->isRunning()) {
    if (checkEStop()) {
      stepper1->forceStop();
      stepper2->forceStop();
      stepper1->setAcceleration(ACCELERATION);
      stepper2->setAcceleration(ACCELERATION);
      return;
    }
    getLogicalPos(logX, logY);
    bool nowSlow = inSlowZone((float)abs(logX), (float)logY);
    if (nowSlow != curSlow) {
      curSlow = nowSlow;
      applySpd(curSlow ? CALIBRATION_SPEED : safeSpd,
               curSlow ? ACCELERATION      : MOVE_ACCEL);
    }
    delay(5);
  }
  stepper1->setAcceleration(ACCELERATION);
  stepper2->setAcceleration(ACCELERATION);
  ESP_LOGI(TAG, "Arrived at %c%c", (int)file, (int)rank);
}


/******** function squareToStepCoords
* Purpose
*   Converts a chess square to its logical step-space X/Y coordinates
*   (float) used by the path-clearance geometry functions.
* Arguments
*   file    File character 'a'-'h' (char, by-value).
*   rank    Rank character '1'-'8' (char, by-value).
*   tx      Reference that receives the X step coordinate (float&).
*   ty      Reference that receives the Y step coordinate (float&).
* Results
*   - Sets tx and ty via reference. No return value.
* Hardware
*   None.
* Software
*   Reads global STEPS_PER_SQUARE constant.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
static void squareToStepCoords(char file, char rank, float &tx, float &ty) {
  tx = (float)(rank - '1')       * STEPS_PER_SQUARE;
  ty = (float)(7 - (file - 'a')) * STEPS_PER_SQUARE;
}


/******** function distToSegSq
* Purpose
*   Returns the squared perpendicular distance from point P to the
*   nearest point on line segment AB. Used by squareInPath() to test
*   whether a square centre is close enough to the electromagnet's
*   travel path to risk accidental piece attraction.
* Arguments
*   px, py    Coordinates of the query point (float, by-value).
*   ax, ay    Coordinates of segment endpoint A (float, by-value).
*   bx, by    Coordinates of segment endpoint B (float, by-value).
* Results
*   - Returns the squared distance (float). No return value.
* Hardware
*   None.
* Software
*   None.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
static float distToSegSq(float px, float py,
                          float ax, float ay,
                          float bx, float by) {
  float dx   = bx - ax;
  float dy   = by - ay;
  float len2 = dx * dx + dy * dy;
  if (len2 < 1.f) {
    float ex = px - ax;
    float ey = py - ay;
    return ex * ex + ey * ey;
  }
  float t = ((px - ax) * dx + (py - ay) * dy) / len2;
  if      (t < 0.f) t = 0.f;
  else if (t > 1.f) t = 1.f;
  float cx = ax + t * dx - px;
  float cy = ay + t * dy - py;
  return cx * cx + cy * cy;
}


/******** function squareInPath
* Purpose
*   Returns true if square (pf, pr) lies within PATH_THRESHOLD_FRAC of
*   one square-width from the straight-line segment between (f1,r1) and
*   (f2,r2) in step-space. Excludes the two endpoint squares themselves.
*   Used to identify pieces that the electromagnet might drag during travel.
* Arguments
*   pf, pr    File/rank of the square to test (char, by-value).
*   f1, r1    Segment start square (char, by-value).
*   f2, r2    Segment end square (char, by-value).
* Results
*   - Returns true if the square centre is within the threshold distance
*     of the segment; false otherwise.
* Hardware
*   None.
* Software
*   Calls squareToStepCoords() and distToSegSq(). Reads PATH_THRESHOLD_FRAC
*   and STEPS_PER_SQUARE constants.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
static bool squareInPath(char pf, char pr, char f1, char r1, char f2, char r2) {
  if (pf == f1 && pr == r1) return false;
  if (pf == f2 && pr == r2) return false;
  float ax, ay, bx, by, px, py;
  squareToStepCoords(f1, r1, ax, ay);
  squareToStepCoords(f2, r2, bx, by);
  squareToStepCoords(pf, pr, px, py);
  float thr = STEPS_PER_SQUARE * PATH_THRESHOLD_FRAC;
  return distToSegSq(px, py, ax, ay, bx, by) < thr * thr;
}


/******** function countNearSegment
* Purpose
*   Counts how many occupied squares on the current pieceBoard[][] are
*   within PATH_THRESHOLD_FRAC of the travel segment from (f1,r1) to
*   (f2,r2). Used by computeMagnetRoute() to score each L-shape option.
* Arguments
*   f1, r1    Segment start square (char, by-value).
*   f2, r2    Segment end square (char, by-value).
* Results
*   - Returns the count of occupied squares near the segment (int).
* Hardware
*   None.
* Software
*   Calls squareInPath(). Reads global pieceBoard[8][8].
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
static int countNearSegment(char f1, char r1, char f2, char r2) {
  int n = 0;
  for (int fi = 0; fi < 8; fi++) {
    for (int ri = 0; ri < 8; ri++) {
      if (pieceBoard[fi][ri] == '.') continue;
      if (squareInPath((char)('a' + fi), (char)('1' + ri), f1, r1, f2, r2)) n++;
    }
  }
  return n;
}


/******** function computeMagnetRoute
* Purpose
*   Determines the optimal path for the electromagnet to carry a piece
*   from (f1,r1) to (f2,r2). For straight moves (same file or rank) the
*   path is always direct. For diagonal moves a direct path is used when
*   no pieces are within the threshold; otherwise the L-shape with the
*   lower total piece-exposure score is chosen. The waypoints written to
*   wpF[]/wpR[] form a sequence of moveToSquare() targets.
* Arguments
*   f1, r1    Source square (char, by-value).
*   f2, r2    Destination square (char, by-value).
*   wpF[]     Output array receiving waypoint file characters (char*).
*   wpR[]     Output array receiving waypoint rank characters (char*).
* Results
*   - Fills wpF[] and wpR[] with 1 or 2 waypoints and returns the count.
* Hardware
*   None.
* Software
*   Calls countNearSegment(). Reads global pieceBoard[8][8].
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
static int computeMagnetRoute(char f1, char r1, char f2, char r2,
                               char wpF[], char wpR[]) {
  if (f1 == f2 || r1 == r2) {
    wpF[0] = f2;
    wpR[0] = r2;
    return 1;
  }

  if (countNearSegment(f1, r1, f2, r2) == 0) {
    wpF[0] = f2;
    wpR[0] = r2;
    return 1;
  }

  char afF  = f2;
  char afR  = r1;
  char bfF  = f1;
  char bfR  = r2;
  bool aFree = (pieceBoard[afF - 'a'][afR - '1'] == '.');
  bool bFree = (pieceBoard[bfF - 'a'][bfR - '1'] == '.');
  int  sA    = countNearSegment(f1, r1, afF, afR) +
               countNearSegment(afF, afR, f2, r2) +
               (aFree ? 0 : 1);
  int  sB    = countNearSegment(f1, r1, bfF, bfR) +
               countNearSegment(bfF, bfR, f2, r2) +
               (bFree ? 0 : 1);
  bool useA;
  if      (sA < sB)         useA = true;
  else if (sB < sA)         useA = false;
  else if (aFree && !bFree) useA = true;
  else if (bFree && !aFree) useA = false;
  else                      useA = true;
  wpF[0] = useA ? afF : bfF;
  wpR[0] = useA ? afR : bfR;
  wpF[1] = f2;
  wpR[1] = r2;
  return 2;
}


/******** function findPathBlockersRoute
* Purpose
*   Finds all pieces that lie in the electromagnet's travel path along
*   the L-shaped route computed by computeMagnetRoute(). Also includes
*   any occupied intermediate waypoint squares. Results are deduplicated.
*   Called by executeMove() to determine which pieces need to be parked
*   before the main move can proceed.
* Arguments
*   f1, r1    Source square (char, by-value).
*   f2, r2    Destination square (char, by-value).
*   bF[]      Output array receiving blocker file characters (char*).
*   bR[]      Output array receiving blocker rank characters (char*).
* Results
*   - Fills bF[] and bR[] with up to MAX_BLOCKERS entries and returns
*     the count of blockers found (int).
* Hardware
*   None.
* Software
*   Calls computeMagnetRoute(), squareInPath(). Reads global pieceBoard.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
static int findPathBlockersRoute(char f1, char r1, char f2, char r2,
                                  char bF[], char bR[]) {
  char wpF[3], wpR[3];
  int  nWP = computeMagnetRoute(f1, r1, f2, r2, wpF, wpR);
  int  n   = 0;
  char pF  = f1;
  char pR  = r1;
  for (int seg = 0; seg < nWP && n < MAX_BLOCKERS; seg++) {
    char nxF = wpF[seg];
    char nxR = wpR[seg];
    for (int fi = 0; fi < 8 && n < MAX_BLOCKERS; fi++) {
      for (int ri = 0; ri < 8 && n < MAX_BLOCKERS; ri++) {
        if (pieceBoard[fi][ri] == '.') continue;
        char f   = (char)('a' + fi);
        char r   = (char)('1' + ri);
        bool dup = false;
        for (int k = 0; k < n; k++) {
          if (bF[k] == f && bR[k] == r) {
            dup = true;
            break;
          }
        }
        if (!dup && squareInPath(f, r, pF, pR, nxF, nxR)) {
          bF[n] = f;
          bR[n] = r;
          n++;
        }
      }
    }
    if (seg < nWP - 1 && pieceBoard[nxF - 'a'][nxR - '1'] != '.') {
      bool dup = false;
      for (int k = 0; k < n; k++) {
        if (bF[k] == nxF && bR[k] == nxR) {
          dup = true;
          break;
        }
      }
      if (!dup && n < MAX_BLOCKERS) {
        bF[n] = nxF;
        bR[n] = nxR;
        n++;
      }
    }
    pF = nxF;
    pR = nxR;
  }
  return n;
}


/******** function findParkingSquare
* Purpose
*   Searches outward from origF/origR in a spiral of increasing Manhattan
*   distance for the nearest empty square that satisfies all constraints:
*   not on the main travel path, not already reserved by resF[]/resR[],
*   and reachable via an L-path whose intermediate waypoint is unoccupied.
*   Used to temporarily relocate captured pieces and route blockers before
*   the main electromagnet move.
* Arguments
*   origF, origR      Square to park from (char, by-value).
*   pathF1, pathR1    Main travel path start (char, by-value).
*   pathF2, pathR2    Main travel path end (char, by-value).
*   resF[], resR[]    Already-reserved squares to avoid (char arrays).
*   nRes              Number of reserved squares (int, by-value).
*   outF, outR        References receiving the chosen parking square (char&).
* Results
*   - Sets outF and outR to the chosen square and returns true if found.
*   - Returns false if no valid parking square exists within 7 squares.
* Hardware
*   None.
* Software
*   Calls squareInPath(). Reads global pieceBoard[8][8].
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
static bool findParkingSquare(char origF, char origR,
                               char pathF1, char pathR1,
                               char pathF2, char pathR2,
                               char resF[], char resR[], int nRes,
                               char &outF, char &outR) {
  int ofi = origF - 'a';
  int ori  = origR - '1';
  for (int dist = 1; dist <= 7; dist++) {
    for (int df = -dist; df <= dist; df++) {
      for (int dr = -dist; dr <= dist; dr++) {
        if (abs(df) != dist && abs(dr) != dist) continue;
        int fi = ofi + df;
        int ri = ori + dr;
        if (fi < 0 || fi >= 8 || ri < 0 || ri >= 8) continue;
        char f = (char)('a' + fi);
        char r = (char)('1' + ri);
        if (pieceBoard[fi][ri] != '.') continue;
        if (squareInPath(f, r, pathF1, pathR1, pathF2, pathR2)) continue;
        bool conflict = false;
        for (int e = 0; e < nRes; e++) {
          if (resF[e] == f && resR[e] == r) {
            conflict = true;
            break;
          }
        }
        if (conflict) continue;
        if (f != origF && r != origR) {
          bool wpAFree = (pieceBoard[f     - 'a'][origR - '1'] == '.');
          bool wpBFree = (pieceBoard[origF - 'a'][r     - '1'] == '.');
          if (!wpAFree && !wpBFree) continue;
        }
        outF = f;
        outR = r;
        return true;
      }
    }
  }
  return false;
}


/******** function physicalPickAndPlace
* Purpose
*   Physically moves a piece from square (f1,r1) to (f2,r2) using the
*   electromagnet. Approaches the source with magnet off, energises it,
*   waits for coupling (400 ms near a wall, 200 ms elsewhere), then
*   carries the piece along the L-shaped route from computeMagnetRoute()
*   and releases at the destination. If isStopped fires during any carry
*   segment the magnet is de-energised and the function returns early.
* Arguments
*   f1, r1    Source square (char, by-value).
*   f2, r2    Destination square (char, by-value).
* Results
*   - Moves the physical piece. Updates isElectromagnetOn and the RGB LED.
*   - Turns magnet off before returning regardless of outcome.
*   - No return value.
* Hardware
*   ELECTROMAGNET_PIN (GPIO 2), RGB LED (GPIO 38), CoreXY steppers.
* Software
*   Calls moveToSquare(), squareToStepCoords(), computeMagnetRoute().
*   Reads globals isStopped, MAX_X_STEPS, MAX_Y_STEPS, SLOWDOWN_ZONE.
*   Writes global isElectromagnetOn.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
static void physicalPickAndPlace(char f1, char r1, char f2, char r2) {
  moveToSquare(f1, r1);
  if (isStopped) return;

  isElectromagnetOn = true;
  digitalWrite(ELECTROMAGNET_PIN, HIGH);
  rgbLedWrite(38, 255, 0, 0);

  float srcTx, srcTy;
  squareToStepCoords(f1, r1, srcTx, srcTy);
  bool nearWall = (srcTx > (float)(MAX_X_STEPS - SLOWDOWN_ZONE) ||
                   srcTy > (float)(MAX_Y_STEPS - SLOWDOWN_ZONE) ||
                   srcTx < (float)SLOWDOWN_ZONE                 ||
                   srcTy < (float)SLOWDOWN_ZONE);
  delay(nearWall ? 600 : 400);

  char wpF[3], wpR[3];
  int  nWP = computeMagnetRoute(f1, r1, f2, r2, wpF, wpR);
  for (int i = 0; i < nWP; i++) {
    moveToSquare(wpF[i], wpR[i]);
    if (isStopped) {
      isElectromagnetOn = false;
      digitalWrite(ELECTROMAGNET_PIN, LOW);
      return;
    }
  }
  isElectromagnetOn = false;
  digitalWrite(ELECTROMAGNET_PIN, LOW);
  rgbLedWrite(38, 255, 140, 0);
  Serial.printf("    [phys] %c%c->%c%c (%dseg%s)\n",
                f1, r1, f2, r2, nWP, nearWall ? " wall" : "");
}


/******** function updatePendingCandidates
* Purpose
*   Re-scans results[][] against legalDests[] every FAST_SCAN_MS while a
*   move is pending confirmation, updating the cycling candidate pool to
*   reflect any newly activated or deactivated legal destination squares.
*   Clears pendingMove if the piece appears to have been returned to its
*   source (source sensor re-activates) or the destination sensor goes low
*   on a non-capture square. No-ops if the source is currently being cycled
*   or if fewer than 6 characters are in pendingMove.
* Arguments
*   None.
* Results
*   - May update cycleCandidates[], numCycleCandidates, cycleIdx,
*     pendingMove, and call lcdUpdateRow1().
*   - May clear pendingMove and reset detector state.
*   - No return value.
* Hardware
*   None directly; reads state derived from muxScanAll() in results[][].
* Software
*   Reads globals pendingMove, numLegalDests, legalDests[], cyclingSource,
*   results[][], pieceBoard[][], cycleCandidates[], numCycleCandidates.
*   Calls lcdUpdateRow1().
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void updatePendingCandidates() {
  if (pendingMove.length() < 6) return;
  if (numLegalDests == 0) return;
  if (cyclingSource) return;

  char   srcF  = pendingMove[0];
  char   srcR  = pendingMove[1];
  String srcSq = String(srcF) + String(srcR);

  String found[MAX_CYCLE_CANDIDATES];
  int    nFound = 0;
  for (int i = 0; i < numLegalDests && nFound < MAX_CYCLE_CANDIDATES; i++) {
    if (legalDests[i] == srcSq) continue;
    int fi = legalDests[i][0] - 'a';
    int ri = legalDests[i][1] - '1';
    int mm = 3 - ri / 2;
    int cc = (ri % 2) * 8 + (7 - fi);
    if (mm < 0 || mm >= NUM_MUX || cc < 0 || cc >= NUM_CHANNELS) continue;
    if (!results[mm][cc]) continue;
    found[nFound++] = legalDests[i];
  }

  if (nFound == 0) {
    char df  = pendingMove[4];
    char dr  = pendingMove[5];
    int  fi  = df - 'a';
    int  ri  = dr - '1';
    int  mm  = 3 - ri / 2;
    int  cc  = (ri % 2) * 8 + (7 - fi);
    bool destStillActive = (mm >= 0 && mm < NUM_MUX &&
                            cc >= 0 && cc < NUM_CHANNELS && results[mm][cc]);
    int  sfi = srcF - 'a';
    int  sri = srcR - '1';
    int  smm = 3 - sri / 2;
    int  scc = (sri % 2) * 8 + (7 - sfi);
    bool srcNowActive  = (smm >= 0 && smm < NUM_MUX &&
                          scc >= 0 && scc < NUM_CHANNELS && results[smm][scc]);
    bool isCaptureDest = (fi >= 0 && fi < 8 && ri >= 0 && ri < 8 &&
                          pieceBoard[fi][ri] != '.');
    if (srcNowActive || (!destStillActive && !isCaptureDest)) {
      Serial.printf("  pendingMove %s cleared (dest low or piece returned)\n",
                    pendingMove.c_str());
      pendingMove        = "";
      detectState        = DETECT_IDLE;
      liftedSquare       = "";
      settleBestSquare   = "";
      settleBestCount    = 0;
      numCycleCandidates = 0;
      cycleIdx           = 0;
      cyclingSource      = false;
      lcdUpdateRow1();
    }
    return;
  }

  bool changed = (nFound != numCycleCandidates);
  if (!changed) {
    for (int i = 0; i < nFound && !changed; i++) {
      bool present = false;
      for (int j = 0; j < numCycleCandidates; j++) {
        if (cycleCandidates[j] == found[i]) {
          present = true;
          break;
        }
      }
      if (!present) changed = true;
    }
  }
  if (!changed) return;

  String prevDst     = pendingMove.substring(4);
  numCycleCandidates = nFound;
  for (int i = 0; i < nFound; i++) cycleCandidates[i] = found[i];

  cycleIdx = 0;
  for (int i = 0; i < nFound; i++) {
    if (cycleCandidates[i] == prevDst) {
      cycleIdx = i;
      break;
    }
  }

  String newPending = srcSq + "->" + cycleCandidates[cycleIdx];
  if (newPending != pendingMove) {
    pendingMove = newPending;
    lcdUpdateRow1();
  }
  Serial.printf("  Candidates updated: %s [%d option%s]\n",
                pendingMove.c_str(), nFound,
                nFound > 1 ? " — cycling every 2s" : "");
}


/******** function physicalMoveViaWaypoint
* Purpose
*   Carries a piece from (f1,r1) through an intermediate square (fm,rm)
*   to (f2,r2) using two sequential physicalPickAndPlace() calls. Updates
*   pieceBoard[][] between steps so that subsequent path-clearance
*   calculations see the correct board state. Returns false immediately if
*   isStopped becomes true at any point.
* Arguments
*   f1, r1    Source square (char, by-value).
*   fm, rm    Intermediate waypoint square (char, by-value).
*   f2, r2    Final destination square (char, by-value).
* Results
*   - Moves piece physically and updates pieceBoard[][] at each step.
*   - Returns true on success; false if isStopped fires mid-move.
* Hardware
*   ELECTROMAGNET_PIN, RGB LED, CoreXY steppers (via physicalPickAndPlace).
* Software
*   Calls physicalPickAndPlace(). Reads/writes global pieceBoard[][]
*   and isStopped.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
static bool physicalMoveViaWaypoint(char f1, char r1,
                                     char fm, char rm,
                                     char f2, char r2) {
  char piece = pieceBoard[f1 - 'a'][r1 - '1'];
  physicalPickAndPlace(f1, r1, fm, rm);
  if (isStopped) return false;
  pieceBoard[fm - 'a'][rm - '1'] = piece;
  pieceBoard[f1 - 'a'][r1 - '1'] = '.';

  physicalPickAndPlace(fm, rm, f2, r2);
  if (isStopped) return false;
  pieceBoard[f2 - 'a'][r2 - '1'] = piece;
  pieceBoard[fm - 'a'][rm - '1'] = '.';
  return true;
}


/******** function executeMove
* Purpose
*   Orchestrates a complete physical chess move for the computer's turn.
*   Handles four common opening knight moves with a dedicated L-path pawn
*   relocation sequence. For all other moves: reserves source, destination,
*   and route waypoints; parks any captured piece; parks route blockers in
*   order; executes the main pick-and-place; moves the rook for castling;
*   restores blockers in reverse order; then updates pieceBoard[][], castling
*   rights, the en-passant target, and currentTurn. Suppresses the move
*   detector throughout and re-syncs boardState[][] from a fresh scan before
*   returning.
* Arguments
*   f1, r1    Source square (char, by-value).
*   f2, r2    Destination square (char, by-value).
* Results
*   - Performs all physical and model updates for one computer move.
*   - Aborts gracefully (suppressDetection reset, lcdUpdate called) if
*     isStopped fires at any point.
*   - No return value.
* Hardware
*   ELECTROMAGNET_PIN, RGB LED (GPIO 38), CoreXY steppers, four MUXes.
* Software
*   Calls physicalMoveViaWaypoint(), physicalPickAndPlace(),
*   computeMagnetRoute(), findParkingSquare(), findPathBlockersRoute(),
*   lmgIsWhite(), muxScanAll(), printBoardState(), lcdUpdate().
*   Reads/writes globals pieceBoard[][], castleWK/WQ/BK/BQ, epFile,
*   epRank, currentTurn, lastDisplayedMove, suppressDetection, isStopped.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void executeMove(char f1, char r1, char f2, char r2) {
  Serial.printf("\nMachine move: %c%c->%c%c\n", f1, r1, f2, r2);
  if (f1 < 'a' || f1 > 'h' || r1 < '1' || r1 > '8' ||
      f2 < 'a' || f2 > 'h' || r2 < '1' || r2 > '8') {
    Serial.println("  Invalid square, aborting.");
    return;
  }

  suppressDetection = true;
  rgbLedWrite(38, 255, 140, 0);

  int  fx1           = f1 - 'a';
  int  ry1           = r1 - '1';
  int  fx2           = f2 - 'a';
  int  ry2           = r2 - '1';
  char movingPiece   = pieceBoard[fx1][ry1];
  char capturedPiece = pieceBoard[fx2][ry2];
  bool isCapture     = (capturedPiece != '.');
  bool handled       = false;

  struct KnightBlock { char nf1, nr1, nf2, nr2, pf, pr, pmf, pmr, pkf, pkr; };
  static const KnightBlock KB[] = {
    {'g','1','f','3', 'g','2','g','3','h','3'},
    {'b','1','c','3', 'b','2','b','3','a','3'},
    {'g','8','f','6', 'g','7','g','6','h','6'},
    {'b','8','c','6', 'b','7','b','6','a','6'},
  };

  for (int k = 0; k < 4; k++) {
    if (f1 != KB[k].nf1 || r1 != KB[k].nr1 ||
        f2 != KB[k].nf2 || r2 != KB[k].nr2) continue;
    if (pieceBoard[KB[k].pf - 'a'][KB[k].pr - '1'] == '.') break;

    Serial.printf("  [knight-L] parking pawn %c%c via %c%c to %c%c\n",
                  KB[k].pf, KB[k].pr, KB[k].pmf, KB[k].pmr,
                  KB[k].pkf, KB[k].pkr);
    if (!physicalMoveViaWaypoint(KB[k].pf,  KB[k].pr,
                                  KB[k].pmf, KB[k].pmr,
                                  KB[k].pkf, KB[k].pkr)) {
      suppressDetection = false;
      return;
    }

    Serial.printf("  [knight-L] moving knight %c%c->%c%c\n", f1, r1, f2, r2);
    physicalPickAndPlace(f1, r1, f2, r2);
    if (isStopped) {
      suppressDetection = false;
      return;
    }
    pieceBoard[fx2][ry2] = movingPiece;
    pieceBoard[fx1][ry1] = '.';

    Serial.printf("  [knight-L] restoring pawn %c%c via %c%c to %c%c\n",
                  KB[k].pkf, KB[k].pkr, KB[k].pmf, KB[k].pmr,
                  KB[k].pf,  KB[k].pr);
    if (!physicalMoveViaWaypoint(KB[k].pkf, KB[k].pkr,
                                  KB[k].pmf, KB[k].pmr,
                                  KB[k].pf,  KB[k].pr)) {
      suppressDetection = false;
      return;
    }

    if (movingPiece == 'K') {
      castleWK = false;
      castleWQ = false;
    }
    if (movingPiece == 'k') {
      castleBK = false;
      castleBQ = false;
    }
    epFile            = -1;
    epRank            = -1;
    currentTurn       = (currentTurn == TURN_WHITE) ? TURN_BLACK : TURN_WHITE;
    lastDisplayedMove = String(f1) + String(r1) + "->" +
                        String(f2) + String(r2);
    Serial.printf("Move done. Turn: %s\n",
                  (currentTurn == TURN_WHITE) ? "WHITE" : "BLACK");
    printBoardState();
    handled = true;
    break;
  }

  if (!handled) {
    char resF[MAX_BLOCKERS + 6];
    char resR[MAX_BLOCKERS + 6];
    int  nRes  = 0;
    resF[nRes] = f1;
    resR[nRes] = r1;
    nRes++;
    resF[nRes] = f2;
    resR[nRes] = r2;
    nRes++;
    char mainWpF[3], mainWpR[3];
    int  nMainWp = computeMagnetRoute(f1, r1, f2, r2, mainWpF, mainWpR);
    for (int i = 0; i < nMainWp - 1; i++) {
      resF[nRes] = mainWpF[i];
      resR[nRes] = mainWpR[i];
      nRes++;
      Serial.printf("  Route wp reserved: %c%c\n", mainWpF[i], mainWpR[i]);
    }

    char capParkF = 0;
    char capParkR = 0;
    if (isCapture) {
      if (findParkingSquare(f2, r2, f1, r1, f2, r2,
                            resF, resR, nRes, capParkF, capParkR)) {
        Serial.printf("  Parking captured %c: %c%c->%c%c\n",
                      capturedPiece, f2, r2, capParkF, capParkR);
        physicalPickAndPlace(f2, r2, capParkF, capParkR);
        if (isStopped) {
          suppressDetection = false;
          return;
        }
        pieceBoard[capParkF - 'a'][capParkR - '1'] = capturedPiece;
        pieceBoard[fx2][ry2]                        = '.';
        resF[nRes] = capParkF;
        resR[nRes] = capParkR;
        nRes++;
      } else {
        Serial.printf("  WARNING: no parking for captured %c at %c%c\n",
                      capturedPiece, f2, r2);
      }
    }

    char bF[MAX_BLOCKERS];
    char bR[MAX_BLOCKERS];
    char pkF[MAX_BLOCKERS];
    char pkR[MAX_BLOCKERS];
    bool parked[MAX_BLOCKERS] = {};
    int  nBlockers = findPathBlockersRoute(f1, r1, f2, r2, bF, bR);
    for (int i = 0; i < nBlockers; i++) {
      if (isStopped) break;
      if (findParkingSquare(bF[i], bR[i], f1, r1, f2, r2,
                            resF, resR, nRes, pkF[i], pkR[i])) {
        Serial.printf("  Parking blocker %c: %c%c->%c%c\n",
                      pieceBoard[bF[i] - 'a'][bR[i] - '1'],
                      bF[i], bR[i], pkF[i], pkR[i]);
        physicalPickAndPlace(bF[i], bR[i], pkF[i], pkR[i]);
        if (isStopped) break;
        pieceBoard[pkF[i] - 'a'][pkR[i] - '1'] = pieceBoard[bF[i] - 'a'][bR[i] - '1'];
        pieceBoard[bF[i]  - 'a'][bR[i]  - '1'] = '.';
        resF[nRes] = pkF[i];
        resR[nRes] = pkR[i];
        nRes++;
        parked[i] = true;
      } else {
        Serial.printf("  WARNING: no parking for blocker at %c%c\n", bF[i], bR[i]);
      }
    }

    if (!isStopped) {
      Serial.printf("  Main: %c%c->%c%c\n", f1, r1, f2, r2);
      physicalPickAndPlace(f1, r1, f2, r2);
    }

    if (!isStopped && toupper(movingPiece) == 'K' && abs(fx2 - fx1) == 2) {
      bool white = lmgIsWhite(movingPiece);
      int  bki   = white ? 0 : 7;
      char bk    = white ? '1' : '8';
      char rook  = white ? 'R' : 'r';
      if (fx2 == 6) {
        physicalPickAndPlace('h', bk, 'f', bk);
        if (!isStopped) {
          pieceBoard[5][bki] = rook;
          pieceBoard[7][bki] = '.';
        }
      } else if (fx2 == 2) {
        physicalPickAndPlace('a', bk, 'd', bk);
        if (!isStopped) {
          pieceBoard[3][bki] = rook;
          pieceBoard[0][bki] = '.';
        }
      }
    }

    for (int i = nBlockers - 1; i >= 0 && !isStopped; i--) {
      if (!parked[i]) continue;
      Serial.printf("  Restoring %c: %c%c->%c%c\n",
                    pieceBoard[pkF[i] - 'a'][pkR[i] - '1'],
                    pkF[i], pkR[i], bF[i], bR[i]);
      physicalPickAndPlace(pkF[i], pkR[i], bF[i], bR[i]);
      if (isStopped) break;
      pieceBoard[bF[i]  - 'a'][bR[i]  - '1'] = pieceBoard[pkF[i] - 'a'][pkR[i] - '1'];
      pieceBoard[pkF[i] - 'a'][pkR[i] - '1'] = '.';
    }

    if (!isStopped) {
      pieceBoard[fx2][ry2] = movingPiece;
      pieceBoard[fx1][ry1] = '.';
      if (isCapture && capParkF != 0) {
        pieceBoard[capParkF - 'a'][capParkR - '1'] = '.';
        Serial.printf("  Captured %c cleared from model at %c%c\n",
                      capturedPiece, capParkF, capParkR);
      }
      if (movingPiece == 'K') {
        castleWK = false;
        castleWQ = false;
      }
      if (movingPiece == 'k') {
        castleBK = false;
        castleBQ = false;
      }
      if (movingPiece == 'R' || capturedPiece == 'R') {
        if (fx1 == 7 || fx2 == 7) castleWK = false;
        if (fx1 == 0 || fx2 == 0) castleWQ = false;
      }
      if (movingPiece == 'r' || capturedPiece == 'r') {
        if (fx1 == 7 || fx2 == 7) castleBK = false;
        if (fx1 == 0 || fx2 == 0) castleBQ = false;
      }
      if (toupper(movingPiece) == 'P' && abs(ry2 - ry1) == 2) {
        epFile = fx1;
        epRank = ry2 + (lmgIsWhite(movingPiece) ? -1 : 1);
      } else {
        epFile = -1;
        epRank = -1;
      }
      currentTurn       = (currentTurn == TURN_WHITE) ? TURN_BLACK : TURN_WHITE;
      lastDisplayedMove = String(f1) + String(r1) + "->" +
                          String(f2) + String(r2);
      Serial.printf("Move done. Turn: %s\n",
                    (currentTurn == TURN_WHITE) ? "WHITE" : "BLACK");
      printBoardState();
    }
  }

  rgbLedWrite(38, 0, 0, 255);
  delay(500);
  muxScanAll();
  memcpy(boardState, results, sizeof(boardState));
  memcpy(rawPrev,    results, sizeof(rawPrev));
  memset(sameCount,  0,       sizeof(sameCount));
  detectState        = DETECT_IDLE;
  pendingMove        = "";
  liftedSquare       = "";
  settleBestSquare   = "";
  settleBestCount    = 0;
  numCycleCandidates = 0;
  cyclingSource      = false;
  suppressDetection  = false;
  lcdUpdate();
}


/******** function homeAndMeasure
* Purpose
*   Homes the CoreXY carriage to the bottom-left corner (h1 = origin) by
*   driving both steppers toward the X limit switch, then toward the Y
*   limit switch at CALIBRATION_SPEED. After both limits are hit, resets
*   both stepper positions to zero. When printMeasurement is true, also
*   reports the measured step counts versus MAX_X_STEPS / MAX_Y_STEPS and
*   flags if the drift exceeds 200 steps (indicating the constants need
*   updating). The e-stop is polled on every iteration of both homing loops.
* Arguments
*   printMeasurement    true = print drift report to Serial after homing;
*                       false = print homing-complete banner only
*                       (bool, by-value).
* Results
*   - Moves carriage to physical (0,0) and calls setCurrentPosition(0)
*     on both steppers. No return value.
* Hardware
*   LIMIT_SWITCH_X_PIN (GPIO 1), LIMIT_SWITCH_Y_PIN (GPIO 4) read as
*   digital inputs. stepper1 and stepper2 driven at CALIBRATION_SPEED.
* Software
*   Calls checkEStop(). Reads globals CALIBRATION_SPEED, ACCELERATION,
*   MAX_X_STEPS, MAX_Y_STEPS, stepper1, stepper2.
* Reference
*   V1.0, Thomas Tsantilas, May 2026
********/
void homeAndMeasure(bool printMeasurement) {
  if (printMeasurement) ESP_LOGI(TAG, "=== RE-HOME + MEASURE ===");
  else                  ESP_LOGI(TAG, "=== AUTO-HOMING ===");

  stepper1->setAcceleration(ACCELERATION);
  stepper2->setAcceleration(ACCELERATION);
  stepper1->setSpeedInHz(CALIBRATION_SPEED);
  stepper2->setSpeedInHz(CALIBRATION_SPEED);
  stepper1->applySpeedAcceleration();
  stepper2->applySpeedAcceleration();

  long startX = stepper1->getCurrentPosition();
  stepper1->runForward();
  stepper2->runForward();
  while (digitalRead(LIMIT_SWITCH_X_PIN) == HIGH) {
    if (checkEStop()) return;
    delay(1);
  }
  stepper1->forceStop();
  stepper2->forceStop();
  long xSteps = abs(stepper1->getCurrentPosition() - startX);
  ESP_LOGI(TAG, "X limit. Steps: %ld", xSteps);
  delay(300);

  stepper1->setSpeedInHz(CALIBRATION_SPEED);
  stepper2->setSpeedInHz(CALIBRATION_SPEED);
  stepper1->applySpeedAcceleration();
  stepper2->applySpeedAcceleration();
  long startY = stepper1->getCurrentPosition();
  stepper1->runBackward();
  stepper2->runForward();
  while (digitalRead(LIMIT_SWITCH_Y_PIN) == HIGH) {
    if (checkEStop()) return;
    delay(1);
  }
  stepper1->forceStop();
  stepper2->forceStop();
  long ySteps = abs(stepper1->getCurrentPosition() - startY);
  ESP_LOGI(TAG, "Y limit. Steps: %ld", ySteps);
  delay(300);

  stepper1->setCurrentPosition(0);
  stepper2->setCurrentPosition(0);

  if (!printMeasurement) {
    Serial.println("\n============================");
    Serial.println("     HOMING COMPLETE        ");
    Serial.println("  h1 = origin (0,0)         ");
    Serial.println("============================\n");
    return;
  }
  Serial.println("\n============================================");
  Serial.println("      RE-HOME + MEASUREMENT COMPLETE       ");
  Serial.printf("  X: %ld steps  expected %d  drift %+ld (%+.2f mm)\n",
                xSteps, MAX_X_STEPS,
                xSteps - (long)MAX_X_STEPS,
                (xSteps - (long)MAX_X_STEPS) * 0.0125f);
  Serial.printf("  Y: %ld steps  expected %d  drift %+ld (%+.2f mm)\n",
                ySteps, MAX_Y_STEPS,
                ySteps - (long)MAX_Y_STEPS,
                (ySteps - (long)MAX_Y_STEPS) * 0.0125f);
  Serial.println("  If |drift|>200 steps update MAX_X/Y_STEPS.");
  Serial.println("  Origin reset to (0,0) = h1.");
  Serial.println("============================================\n");
}
