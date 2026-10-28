#include <cstddef>
#include <iostream>
#include "../licenses/types_c.h"
int main() {
    static_assert(sizeof(sl::StreamingParameters) == 100);
    static_assert(offsetof(sl::StreamingParameters, port) == 44);
    static_assert(offsetof(sl::StreamingParameters, input_format) == 60);
    static_assert(offsetof(sl::StreamingParameters, depth_bitrate) == 96);
    std::cout << "Official header ABI: size 100, offsets 44/60/96 OK\n";
}
