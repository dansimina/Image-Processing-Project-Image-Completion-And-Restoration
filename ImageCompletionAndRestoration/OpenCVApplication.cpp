// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <random>

const int PATCH_SIZE = 9;
const int DELTA = PATCH_SIZE / 2;

wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

// Structure to hold selection data
struct SelectionData {
	Mat original;
	bool update = false;
	bool selected = false;
	int startX = 0;
	int startY = 0;
	int endX = 0;
	int endY = 0;
	int minX = 0;
	int minY = 0;
	int maxX = 0;
	int maxY = 0;
};

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	SelectionData* data = (SelectionData*)param;
	Mat img = data->original.clone(); 

	if (event == EVENT_LBUTTONDOWN && !data->selected)
	{
		if (x >= data->minX && x <= data->maxX && y >= data->minY && y <= data->maxY) {
			data->update = true;
			data->startX = x;
			data->startY = y;
			printf("StartPos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)data->original.at<Vec3b>(y, x)[2],
				(int)data->original.at<Vec3b>(y, x)[1],
				(int)data->original.at<Vec3b>(y, x)[0]);
		}
	}
	else if (event == EVENT_MOUSEMOVE && data->update)
	{
		if (x >= data->minX && x <= data->maxX && y >= data->minY && y <= data->maxY) {
			data->endX = x;
			data->endY = y;

			img = data->original.clone();

			rectangle(img, Point(data->startX, data->startY),
				Point(data->endX, data->endY),
				Scalar(0, 255, 0), 2);

			imshow("My Window", img);
		}

		//printf("IntPos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
		//	x, y,
		//	(int)data->original.at<Vec3b>(y, x)[2],
		//	(int)data->original.at<Vec3b>(y, x)[1],
		//	(int)data->original.at<Vec3b>(y, x)[0]);
	}
	else if (event == EVENT_LBUTTONUP && data->update)
	{
		if (x >= data->minX && x <= data->maxX && y >= data->minY && y <= data->maxY) {
			data->update = false;
			data->selected = true;
			data->endX = x;
			data->endY = y;

			img = data->original.clone();

			rectangle(img, Point(data->startX, data->startY),
				Point(data->endX, data->endY),
				Scalar(0, 255, 0), 2);

			imshow("My Window", img);

			printf("EndPos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)data->original.at<Vec3b>(y, x)[2],
				(int)data->original.at<Vec3b>(y, x)[1],
				(int)data->original.at<Vec3b>(y, x)[0]);
		}
	}
}

std::vector<std::vector<bool>> computeMask(Mat img, int startX, int startY, int endX, int endY) {
	std::vector<std::vector<bool>> mask(img.rows, std::vector<bool>(img.cols, false));

	startX = max(DELTA, startX);
	startY = max(DELTA, startY);
	endX = min(img.cols - 1 - DELTA, endX);
	endY = min(img.rows - 1 - DELTA, endY);

	for (int i = startY; i <= endY; i++) {
		for (int j = startX; j <= endX; j++) {
			mask[i][j] = true;
		}
	}

	return mask;
}

std::vector<std::pair<int, int>> generateRandomPairs(const std::vector<std::vector<bool>>& mask, int startX, int endX,
	int startY, int endY, int numPairs) {
	if (mask.empty() || mask[0].empty()) {
		return {};
	}

	if (startX > endX || startY > endY) {
		return {};
	}

	int maxPairs = (endX - startX + 1) * (endY - startY + 1);
	numPairs = min(numPairs, maxPairs);

	if (numPairs <= 0) {
		return {};
	}

	std::vector<std::pair<int, int>> result;
	std::set<std::pair<int, int>> usedPairs;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distX(startX, endX);
	std::uniform_int_distribution<> distY(startY, endY);

	int attempts = 0;
	const int MAX_ATTEMPTS = 2000;

	while (result.size() < numPairs && attempts < MAX_ATTEMPTS) {
		int x = distX(gen);
		int y = distY(gen);

		bool valid = true;
		for (int dy = -DELTA; dy <= DELTA && valid; dy++) {
			for (int dx = -DELTA; dx <= DELTA && valid; dx++) {
				int nx = x + dx;
				int ny = y + dy;
				if (nx < 0 || nx >= mask[0].size() || ny < 0 || ny >= mask.size() || mask[ny][nx]) {
					valid = false;
				}
			}
		}

		if (valid) {
			std::pair<int, int> newPair = { x, y };
			if (usedPairs.find(newPair) == usedPairs.end()) {
				result.push_back(newPair);
				usedPairs.insert(newPair);
			}
		}
		attempts++;
	}

	return result;
}

double computeMSE(const Mat& img, const std::vector<std::vector<bool>>& mask,
	int x1, int y1, int x2, int y2) {
	double sum = 0;
	int cnt = 0;

	for (int dy = -DELTA; dy <= DELTA; dy++) {
		for (int dx = -DELTA; dx <= DELTA; dx++) {
			int y1d = y1 + dy;
			int x1d = x1 + dx;
			int y2d = y2 + dy;
			int x2d = x2 + dx;

			if (y1d >= 0 && y1d < img.rows && x1d >= 0 && x1d < img.cols &&
				y2d >= 0 && y2d < img.rows && x2d >= 0 && x2d < img.cols &&
				!mask[y1d][x1d]) {

				Vec3b p1 = img.at<Vec3b>(y1d, x1d);
				Vec3b p2 = img.at<Vec3b>(y2d, x2d);

				for (int i = 0; i < 3; i++) {
					int diff = abs(p1[i] - p2[i]);
					sum += diff * diff;
				}
				
				cnt++;
			}
		}
	}

	return cnt > 0 ? sum / cnt : DBL_MAX;
}

std::pair<int, int> findBestMatch(const Mat& img, const std::vector<std::vector<bool>>& mask,
	int x, int y) {
	int startX = DELTA;
	int endX = img.cols - 1 - DELTA;
	int startY = DELTA;
	int endY = img.rows - 1 - DELTA;

	double bestMatch = DBL_MAX;
	int bestX = startX, bestY = startY;

	int n = (img.rows * img.cols) / 4;

	std::vector<std::pair<int, int>> offsets = generateRandomPairs(mask, startX, endX, startY, endY, n);

	for (const auto& offset : offsets) {
		double diff = computeMSE(img, mask, x, y, offset.first, offset.second);
		if (diff < bestMatch) {
			bestMatch = diff;
			bestX = offset.first;
			bestY = offset.second;
		}
	}

	for (int i = 0; i < 5; i++) {
		int width = (endX - startX) / 2;
		int height = (endY - startY) / 2;

		n /= 2;

		startX = max(DELTA, bestX - width);
		startY = max(DELTA, bestY - height);
		endX = min(img.cols - 1 - DELTA, bestX + width);
		endY = min(img.rows - 1 - DELTA, bestY + height);

		std::vector<std::pair<int, int>> newOffsets = generateRandomPairs(mask, startX, endX, startY, endY, 100);

		for (const auto& offset : newOffsets) {
			double diff = computeMSE(img, mask, x, y, offset.first, offset.second);
			if (diff < bestMatch) {
				bestMatch = diff;
				bestX = offset.first;
				bestY = offset.second;
			}
		}
	}

	return { bestX, bestY };
}

void completePatch(Mat& img, std::vector<std::vector<bool>>& mask, int x, int y) {
	std::pair<int, int> offset = findBestMatch(img, mask, x, y);

	for (int dy = -DELTA; dy <= DELTA; dy++) {
		for (int dx = -DELTA; dx <= DELTA; dx++) {
			int targetY = y + dy;
			int targetX = x + dx;
			int sourceY = offset.second + dy;
			int sourceX = offset.first + dx;

			if (targetY >= 0 && targetY < img.rows && targetX >= 0 && targetX < img.cols &&
				sourceY >= 0 && sourceY < img.rows && sourceX >= 0 && sourceX < img.cols) {

				if (mask[targetY][targetX]) {
					img.at<Vec3b>(targetY, targetX) = img.at<Vec3b>(sourceY, sourceX);
					mask[targetY][targetX] = false;
				}
				else {
					int d = max(abs(dy), abs(dx));
					double weight = d / (double)DELTA;
					img.at<Vec3b>(targetY, targetX)[0] = img.at<Vec3b>(sourceY, sourceX)[0] * (1 - weight) + img.at<Vec3b>(targetY, targetX)[0] * weight;
					img.at<Vec3b>(targetY, targetX)[1] = img.at<Vec3b>(sourceY, sourceX)[1] * (1 - weight) + img.at<Vec3b>(targetY, targetX)[1] * weight;
					img.at<Vec3b>(targetY, targetX)[2] = img.at<Vec3b>(sourceY, sourceX)[2] * (1 - weight) + img.at<Vec3b>(targetY, targetX)[2] * weight;
				}
			}
		}
	}
}

Mat imageReconstruction(Mat& img, int startX, int startY, int endX, int endY)
{
	Mat reconstruction = img.clone();

	startX = max(DELTA, startX);
	startY = max(DELTA, startY);
	endX = min(img.cols - 1 - DELTA, endX);
	endY = min(img.rows - 1 - DELTA, endY);

	if (endX - startX + 1 > 0 (endX - startX + 1) % PATCH_SIZE == 0) {
		endX--;
	}

	if (endY - startY + 1 > 0 (endY - startY + 1) % PATCH_SIZE == 0) {
		endY--;
	}

	std::vector<std::vector<bool>> mask = computeMask(img, startX, startY, endX, endY);

	for (int y = startY + 1; y <= endY - 1; y++) {
		for (int x = startX + 1; x <= endX - 1; x++) {
			if (mask[y][x]) {
				reconstruction.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
			}
		}
	}
	imshow("Region to Fill", reconstruction);
	waitKey(0);

	std::queue<std::pair<int, int>> Q;
	Q.push({ startX + 2, startY + 2 });
	Q.push({ startX + 2, endY - 2 });
	Q.push({ endX - 2, startY + 2 });
	Q.push({ endX - 2, endY - 2 });

	while (!Q.empty()) {
		auto front = Q.front();
		Q.pop();
		int x = front.first;
		int y = front.second;

		if (x >= startX && y >= startY && x <= endX && y <= endY && mask[y][x]) {
			completePatch(reconstruction, mask, x, y);

			Q.push({ x - DELTA - 1, y });
			Q.push({ x, y - DELTA - 1 });
			Q.push({ x + DELTA + 1, y });
			Q.push({ x, y + DELTA + 1 });
		}
	}

	for (int y = startY; y <= endY; y++) {
		for (int x = startX; x <= endX; x++) {
			if (mask[y][x]) {
				completePatch(reconstruction, mask, x, y);
			}
		}
	}

	return reconstruction;
}

void testMouseClick()
{
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		// Set up the selection data
		SelectionData data;
		data.original = imread(fname);
		data.minX = PATCH_SIZE / 2;
		data.minY = PATCH_SIZE / 2;
		data.maxX = data.original.cols - PATCH_SIZE / 2 - 1;
		data.maxY = data.original.rows - PATCH_SIZE / 2 - 1;

		// Create a window
		namedWindow("My Window", 1);

		// Set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &data);

		// Show the original image
		imshow("My Window", data.original);

		// Wait until user press some key
		waitKey(0);

		if (data.selected) {
			int startX = min(data.startX, data.endX);
			int startY = min(data.startY, data.endY);
			int endX = max(data.startX, data.endX);
			int endY = max(data.startY, data.endY);

			Mat img = imageReconstruction(data.original, startX, startY, endX, endY);

			imshow("Selected Region", img);
			waitKey(0);
		}
	}
}


int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
	projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 6 - BGR->Gray (fast, save result to disk) \n");
		printf(" 8 - Resize image\n");
		printf(" 12 - Mouse callback demo\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 6:
			testImageOpenAndSave();
			break;
		case 8:
			testResize();
			break;
		case 12:
			testMouseClick();
			break;
		}
	} while (op != 0);
	return 0;
}