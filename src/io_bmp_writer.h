#ifndef IO_BMP_WRITER_H_
#define IO_BMP_WRITER_H_

#include <stdint.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>

// resource used:
// https://dev.to/muiz6/c-how-to-write-a-bitmap-image-from-scratch-1k6m
// https://en.wikipedia.org/wiki/BMP_file_format

// pack the struct tightly with no word-alignment but byte for byte. this also
// enables `sizeof` to work as expected (14UL for BitmapFileHeader, not 16UL)
#pragma pack(push, 1)
/// @brief Implementation of the Windows BITMAPINFOHEADER standard
struct BitmapInformationHeader {
    uint32_t own_size { sizeof(*this) }; // should be 40UL
    int32_t width; // set by constructor
    int32_t height; // set by constructor
    uint16_t colour_plane_count { 1 };
    uint16_t bits_per_pixel { 24 };
    uint32_t compression { 0 }; // 0 indicates no compression
    uint32_t dummy_bitmap_size { 0 }; // ignore
    // 300DPI is roughly 11810px/m:
    int32_t horizontal_px_per_m { 11810 }; // signed px/m resolution
    int32_t vertical_px_per_m { 11810 }; // signed px/m resolution
    uint32_t colour_table_entries { 0 }; // zero defaults to 2^n
    uint32_t important_colours_count {
        0
    }; // ignore: every colour is important :)

    ///@brief Construct a new Bitmap Information Header object
    ///
    ///@param _width width of the image to construct
    ///@param _height height of the image to construct
    BitmapInformationHeader(const uint _width, const uint _height)
        : width((uint32_t)_width)
        , height((uint32_t)_height) {};
};
#pragma pack(pop)

#pragma pack(push, 1)
struct BitmapFileHeader {
    char header_field[2] = { 'B', 'M' };
    uint32_t file_size; // set by constructor
    uint32_t reserved { 0 }; // 4 reserved bytes
    uint32_t pixel_data_offset { sizeof(*this)
        + sizeof(BitmapInformationHeader) };

    ///@brief Construct a new Bmp Header object
    ///
    ///@param width width of the image to construct
    ///@param height height of the image to construct
    BitmapFileHeader(const uint width, const uint height)
        : file_size(pixel_data_offset
              + ((uint32_t)width) * ((uint32_t)height) * 3) {};
};
#pragma pack(pop)

void write_bmp(std::vector<unsigned char>& data, const uint width,
    const uint height, const std::filesystem::path& path)
{
    // open file output stream
    std::ofstream out(path, std::ios::out | std::ios::binary);

    // create BMP header and info segments
    const BitmapFileHeader header(width, height);
    const BitmapInformationHeader info(width, height);
    // write header and info segments
    out.write(reinterpret_cast<const char*>(&header), sizeof(BitmapFileHeader));
    out.write(
        reinterpret_cast<const char*>(&info), sizeof(BitmapInformationHeader));
    // then write the entire contents of the vector
    out.write(reinterpret_cast<const char*>(data.data()),
        static_cast<std::streamsize>(data.size()));
}

#endif // IO_BMP_WRITER_H_