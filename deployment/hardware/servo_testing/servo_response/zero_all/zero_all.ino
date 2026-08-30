#include <Servo.h>

Servo srv;

void setup() {
    srv.writeMicroseconds(1500);
    srv.attach(9, 500, 2500);
}

void loop() {}
