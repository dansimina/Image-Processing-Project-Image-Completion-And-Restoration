// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <random>
#include <thread>
#include <mutex>
#include <unordered_set>

const int PATCH_SIZE = 13;
const int DELTA = PATCH_SIZE / 2;
const int STEP = DELTA / 2;

wchar_t* projectPath;

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

struct PairHash {
	template <class T1, class T2>
	std::size_t operator() (const std::pair<T1, T2>& pair) const {
		auto hash1 = std::hash<T1>{}(pair.first);
		auto hash2 = std::hash<T2>{}(pair.second);
		return hash1 * 31 + hash2; 
	}
};

bool isValidPoint(const std::vector<std::vector<bool>>& mask, int x, int y) {
	const std::vector<std::pair<int, int>> cornerOffsets = {
		{-DELTA, -DELTA}, { DELTA, -DELTA},
		{-DELTA,  DELTA}, { DELTA,  DELTA}
	};

	for (const auto& offset : cornerOffsets) {
		int cornerX = x + offset.first;
		int cornerY = y + offset.second;
		if (mask[cornerY][cornerX]) {
			return false;
		}
	}
	return true;
}

std::vector<std::pair<int, int>> generateRandomPairs(
	const std::vector<std::vector<bool>>& mask,
	int searchStartX, int searchEndX,
	int searchStartY, int searchEndY,
	int numPairs)
{
	if (mask.empty() || mask[0].empty() || searchStartX > searchEndX || searchStartY > searchEndY || numPairs <= 0) {
		return {};
	}

	const int maskHeight = mask.size();
	const int maskWidth = mask[0].size();
	const int searchWidth = searchEndX - searchStartX + 1;
	const int searchHeight = searchEndY - searchStartY + 1;

	if (searchWidth <= 0 || searchHeight <= 0) {
		return {};
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	

	std::vector<std::pair<int, int>> result;
	result.reserve(numPairs);
	std::unordered_set<std::pair<int, int>, PairHash> generatedPairs;

	const int gridSize = 5;
	const double cellWidth = searchWidth / gridSize;
	const double cellHeight = searchHeight / gridSize;
	const int maxAttemptsPerCell = min(cellWidth * cellHeight,  100);
	const int pairsPerCell = numPairs / (gridSize * gridSize);

	if (pairsPerCell > 0 && maxAttemptsPerCell > 0) {
		for (int gridY = 0; gridY < gridSize && result.size() < numPairs; gridY++) {
			for (int gridX = 0; gridX < gridSize && result.size() < numPairs; gridX++) {
				int cellStartX = searchStartX + gridX * cellWidth;
				int cellEndX = min(searchEndX, searchStartX + (gridX + 1) * cellWidth) - 1;
				int cellStartY = searchStartY + gridY * cellHeight;
				int cellEndY = min(searchEndY, searchStartY + (gridY + 1) * cellHeight) - 1;

				if (cellStartX > cellEndX || cellStartY > cellEndY) continue;

				std::uniform_int_distribution<> distribX(cellStartX, cellEndX);
				std::uniform_int_distribution<> distribY(cellStartY, cellEndY);

				for (int attempt = 0, cnt = 0; cnt < pairsPerCell && attempt < maxAttemptsPerCell; attempt++) {
					int x = distribX(gen);
					int y = distribY(gen);

					if (isValidPoint(mask, x, y)) {
						std::pair<int, int> currentPair = { x, y };
						if (generatedPairs.find(currentPair) == generatedPairs.end()) {
							result.push_back(currentPair);
							generatedPairs.insert(currentPair);
							cnt++;
						}
					}
				}
			}
		}
	}

	if (result.size() < numPairs) {
		std::uniform_int_distribution<> distribX(searchStartX, searchEndX);
		std::uniform_int_distribution<> distribY(searchStartY, searchEndY);

		const int remainingPairs = numPairs - result.size();
		const long long searchArea = (long long)(searchWidth) * searchHeight;
		const int maxFallbackAttempts = max(remainingPairs * 50, 1000);


		for (int attempt = 0; attempt < maxFallbackAttempts && result.size() < numPairs; attempt++) {
			int x = distribX(gen);
			int y = distribY(gen);

			if (isValidPoint(mask, x, y)) {
				std::pair<int, int> currentPair = { x, y };
				if (generatedPairs.find(currentPair) == generatedPairs.end()) {
					result.push_back(currentPair);
					generatedPairs.insert(currentPair);
				}
			}
		}
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

			if (!mask[y1d][x1d]) {
				Vec3b pixel1 = img.at<Vec3b>(y1d, x1d);
				Vec3b pixel2 = img.at<Vec3b>(y2d, x2d);

				for (int i = 0; i < 3; i++) {
					int diff = abs(pixel1[i] - pixel2[i]);
					sum += diff * diff;
				}

				cnt++;
			}
		}
	}

	return cnt > 0 ? sum / cnt : DBL_MAX;
}

std::pair<int, int> findBestMatch(const Mat& img, const std::vector<std::vector<bool>>& mask, int x, int y) {
	int startX = DELTA;
	int endX = img.cols - 1 - DELTA;
	int startY = DELTA;
	int endY = img.rows - 1 - DELTA;

	double bestMatch = DBL_MAX;
	int bestX = startX, bestY = startY;

	int n = std::sqrt(img.rows * img.cols) / 2;

	std::vector<std::pair<int, int>> offsets = generateRandomPairs(mask, startX, endX, startY, endY, n);

	for (const auto& offset : offsets) {
		double diff = computeMSE(img, mask, x, y, offset.first, offset.second);
		if (diff < bestMatch) {
			bestMatch = diff;
			bestX = offset.first;
			bestY = offset.second;

			if (diff < 25.0) {
				return { bestX, bestY };
			}
		}
	}

	for (int i = 0; i < 5; i++) {
		int width = (endX - startX) / 2;
		int height = (endY - startY) / 2;

		n = std::sqrt(height * width);

		startX = max(DELTA, bestX - width);
		startY = max(DELTA, bestY - height);
		endX = min(img.cols - 1 - DELTA, bestX + width);
		endY = min(img.rows - 1 - DELTA, bestY + height);

		offsets = generateRandomPairs(mask, startX, endX, startY, endY, n);

		for (const auto& offset : offsets) {
			double diff = computeMSE(img, mask, x, y, offset.first, offset.second);
			if (diff < bestMatch) {
				bestMatch = diff;
				bestX = offset.first;
				bestY = offset.second;

				if (diff < 25.0) {
					return { bestX, bestY };
				}
			}
		}
	}

	return { bestX, bestY };
}

void completePatch(Mat& img, std::vector<std::vector<bool>>& mask, int x, int y) {
	std::pair<int, int> offset = findBestMatch(img, mask, x, y);

	for (int dy = -STEP; dy <= STEP; dy++) {
		for (int dx = -STEP; dx <= STEP; dx++) {
			int targetY = y + dy;
			int targetX = x + dx;
			int sourceY = offset.second + dy;
			int sourceX = offset.first + dx;

			if (mask[targetY][targetX]) {
				img.at<Vec3b>(targetY, targetX) = img.at<Vec3b>(sourceY, sourceX);
				mask[targetY][targetX] = false;
			}
		}
	}

	for (int dy = -DELTA; dy <= DELTA; dy++) {
		for (int dx = -DELTA; dx <= DELTA; dx++) {
			if (abs(dx) >= STEP || abs(dy) >= STEP) {
				int targetY = y + dy;
				int targetX = x + dx;
				int sourceY = offset.second + dy;
				int sourceX = offset.first + dx;

				if (!mask[targetY][targetX]) {
					// Calculate distance from patch center
					double dist = sqrt(dx * dx + dy * dy) / DELTA;

					// Calculate weight with smoother falloff
					double weight = pow(dist, 1.5); // Adjust exponent for transition sharpness
					weight = min(1.0, max(0.0, weight));

					// Get gradient information
					Vec3b sourcePixel = img.at<Vec3b>(sourceY, sourceX);
					Vec3b targetPixel = img.at<Vec3b>(targetY, targetX);

					// Calculate gradient-aware blending
					for (int c = 0; c < 3; c++) {
						// Blend with respect to local gradient
						img.at<Vec3b>(targetY, targetX)[c] =
							sourcePixel[c] * (1 - weight) + targetPixel[c] * weight;
					}
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
	Q.push({ startX + 1, startY + 1 });
	Q.push({ endX - 1, startY + 1 });
	Q.push({ startX + 1, endY - 1 });
	Q.push({ endX - 1, endY - 1 });

	while (!Q.empty()) {
		auto front = Q.front();
		Q.pop();
		int x = front.first;
		int y = front.second;

		if (mask[y][x] && x >= startX && y >= startY && x <= endX && y <= endY) {
			completePatch(reconstruction, mask, x, y);

			if (x - STEP - 1 >= 0 && mask[y][x - STEP - 1]) {
				Q.push({ x - STEP - 1, y });
			}
			if (y - STEP - 1 >= 0 && mask[y - STEP - 1][x]) {
				Q.push({ x, y - STEP - 1 });
			}
			if (x + STEP + 1 < reconstruction.cols && mask[y][x + STEP + 1]) {
				Q.push({ x + STEP + 1, y });
			}
			if (y + STEP + 1 < reconstruction.rows && mask[y + STEP + 1][x]) {
				Q.push({ x, y + STEP + 1 });
			}
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
		printf(" 1 - Demo\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testMouseClick();
			break;
		}
	} while (op != 0);
	return 0;
}