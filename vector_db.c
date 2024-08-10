#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <pthread.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#define VECTOR_DIM 100
#define INDEX_SIZE 1000000000ULL  // 1 billion vectors
#define CHUNK_SIZE 1000000        // Number of vectors per chunk
#define NUM_CHUNKS (INDEX_SIZE / CHUNK_SIZE)
#define MAX_THREADS 16
#define HNSW_M 16                 // Number of connections per layer in HNSW
#define HNSW_EF_CONSTRUCTION 200  // Size of dynamic candidate list for HNSW construction

typedef struct {
    float coordinates[VECTOR_DIM];
    int id;
} Vector;

typedef struct HNSWNode {
    Vector vector;
    struct HNSWNode** neighbors;
    int* num_neighbors;
    int max_level;
} HNSWNode;

typedef struct {
    HNSWNode** layers;
    int num_layers;
    int max_elements;
    int cur_element;
} HNSWIndex;

typedef struct {
    Vector* vectors;
    int size;
    int fd;
    char filename[256];
} Chunk;

typedef struct {
    Chunk* chunks[NUM_CHUNKS];
    int num_chunks;
    HNSWIndex* hnsw_index;
} VectorDB;

// Function prototypes
VectorDB* createVectorDB();
void insertVector(VectorDB* db, Vector* vector);
Vector* findNearest(VectorDB* db, Vector* query);
void freeVectorDB(VectorDB* db);
void generateRandomVector(Vector* v);
float distance(Vector* a, Vector* b);

// HNSW Index functions
HNSWIndex* createHNSWIndex(int max_elements);
void insertHNSW(HNSWIndex* index, Vector* vector);
Vector* searchHNSW(HNSWIndex* index, Vector* query, int ef);
void freeHNSWIndex(HNSWIndex* index);

// Memory-mapped file functions
Chunk* createChunk(int chunk_id);
void writeVectorToChunk(Chunk* chunk, Vector* vector);
Vector* readVectorFromChunk(Chunk* chunk, int index);

// Multithreading
typedef struct {
    VectorDB* db;
    Vector* query;
    int start_chunk;
    int end_chunk;
    Vector* result;
    float min_dist;
} ThreadArgs;

void* searchChunkRange(void* args);

// Main VectorDB functions
VectorDB* createVectorDB() {
    VectorDB* db = (VectorDB*)malloc(sizeof(VectorDB));
    db->num_chunks = 0;
    for (int i = 0; i < NUM_CHUNKS; i++) {
        db->chunks[i] = NULL;
    }
    db->hnsw_index = createHNSWIndex(INDEX_SIZE);
    return db;
}

void insertVector(VectorDB* db, Vector* vector) {
    int chunk_index = db->num_chunks - 1;
    if (chunk_index < 0 || db->chunks[chunk_index]->size >= CHUNK_SIZE) {
        if (db->num_chunks >= NUM_CHUNKS) {
            printf("Database is full\n");
            return;
        }
        db->chunks[db->num_chunks] = createChunk(db->num_chunks);
        chunk_index = db->num_chunks;
        db->num_chunks++;
    }
    
    Chunk* chunk = db->chunks[chunk_index];
    writeVectorToChunk(chunk, vector);
    insertHNSW(db->hnsw_index, vector);
}

Vector* findNearest(VectorDB* db, Vector* query) {
    // Use HNSW for approximate nearest neighbor search
    return searchHNSW(db->hnsw_index, query, HNSW_EF_CONSTRUCTION);
}

void freeVectorDB(VectorDB* db) {
    for (int i = 0; i < db->num_chunks; i++) {
        close(db->chunks[i]->fd);
        unlink(db->chunks[i]->filename);
        free(db->chunks[i]);
    }
    freeHNSWIndex(db->hnsw_index);
    free(db);
}

// HNSW Index implementation
HNSWIndex* createHNSWIndex(int max_elements) {
    HNSWIndex* index = (HNSWIndex*)malloc(sizeof(HNSWIndex));
    index->max_elements = max_elements;
    index->cur_element = 0;
    index->num_layers = 1 + (int)(log(max_elements) / log(HNSW_M));
    index->layers = (HNSWNode**)malloc(index->num_layers * sizeof(HNSWNode*));
    for (int i = 0; i < index->num_layers; i++) {
        index->layers[i] = (HNSWNode*)calloc(max_elements, sizeof(HNSWNode));
    }
    return index;
}

void insertHNSW(HNSWIndex* index, Vector* vector) {
    // Simplified HNSW insertion
    int insert_level = index->cur_element == 0 ? index->num_layers - 1 : 
                       (int)((-log(drand48()) * HNSW_M) / log(1.0 / HNSW_M));
    
    for (int level = 0; level <= insert_level; level++) {
        HNSWNode* node = &index->layers[level][index->cur_element];
        node->vector = *vector;
        node->neighbors = (HNSWNode**)malloc(HNSW_M * sizeof(HNSWNode*));
        node->num_neighbors = (int*)calloc(1, sizeof(int));
        node->max_level = insert_level;
    }
    
    index->cur_element++;
}

Vector* searchHNSW(HNSWIndex* index, Vector* query, int ef) {
    // Simplified HNSW search
    int top_layer = index->num_layers - 1;
    HNSWNode* curr = &index->layers[top_layer][0];
    
    for (int level = top_layer; level > 0; level--) {
        HNSWNode* next_curr = NULL;
        float min_dist = FLT_MAX;
        
        for (int i = 0; i < *curr->num_neighbors; i++) {
            float dist = distance(&curr->neighbors[i]->vector, query);
            if (dist < min_dist) {
                min_dist = dist;
                next_curr = curr->neighbors[i];
            }
        }
        
        if (next_curr) curr = next_curr;
    }
    
    // On the bottom layer, perform a more thorough search
    float min_dist = FLT_MAX;
    Vector* nearest = NULL;
    
    for (int i = 0; i < index->cur_element; i++) {
        float dist = distance(&index->layers[0][i].vector, query);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = &index->layers[0][i].vector;
        }
    }
    
    return nearest;
}

void freeHNSWIndex(HNSWIndex* index) {
    for (int i = 0; i < index->num_layers; i++) {
        for (int j = 0; j < index->cur_element; j++) {
            free(index->layers[i][j].neighbors);
            free(index->layers[i][j].num_neighbors);
        }
        free(index->layers[i]);
    }
    free(index->layers);
    free(index);
}

// Memory-mapped file functions
Chunk* createChunk(int chunk_id) {
    Chunk* chunk = (Chunk*)malloc(sizeof(Chunk));
    sprintf(chunk->filename, "chunk_%d.bin", chunk_id);
    chunk->fd = open(chunk->filename, O_CREAT | O_RDWR, 0666);
    if (chunk->fd == -1) {
        perror("Error opening file");
        exit(1);
    }
    
    off_t file_size = CHUNK_SIZE * sizeof(Vector);
    if (lseek(chunk->fd, file_size - 1, SEEK_SET) == -1) {
        perror("Error calling lseek() to 'stretch' the file");
        exit(1);
    }
    if (write(chunk->fd, "", 1) == -1) {
        perror("Error writing last byte of the file");
        exit(1);
    }
    
    chunk->vectors = (Vector*)mmap(0, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, chunk->fd, 0);
    if (chunk->vectors == MAP_FAILED) {
        perror("Error mmapping the file");
        exit(1);
    }
    
    chunk->size = 0;
    return chunk;
}

void writeVectorToChunk(Chunk* chunk, Vector* vector) {
    if (chunk->size < CHUNK_SIZE) {
        memcpy(&chunk->vectors[chunk->size], vector, sizeof(Vector));
        chunk->size++;
    }
}

Vector* readVectorFromChunk(Chunk* chunk, int index) {
    if (index < chunk->size) {
        return &chunk->vectors[index];
    }
    return NULL;
}

// Multithreading functions
void* searchChunkRange(void* args) {
    ThreadArgs* thread_args = (ThreadArgs*)args;
    Vector* local_nearest = NULL;
    float local_min_dist = FLT_MAX;
    
    for (int i = thread_args->start_chunk; i < thread_args->end_chunk; i++) {
        Chunk* chunk = thread_args->db->chunks[i];
        for (int j = 0; j < chunk->size; j++) {
            Vector* v = readVectorFromChunk(chunk, j);
            float dist = distance(v, thread_args->query);
            if (dist < local_min_dist) {
                local_min_dist = dist;
                local_nearest = v;
            }
        }
    }
    
    thread_args->result = local_nearest;
    thread_args->min_dist = local_min_dist;
    return NULL;
}

// Utility functions
void generateRandomVector(Vector* v) {
    for (int i = 0; i < VECTOR_DIM; i++) {
        v->coordinates[i] = (float)rand() / RAND_MAX;
    }
    v->id = rand();
}

float distance(Vector* a, Vector* b) {
    float sum = 0;
    for (int i = 0; i < VECTOR_DIM; i++) {
        float diff = a->coordinates[i] - b->coordinates[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

int main() {
    srand(time(NULL));
    
    VectorDB* db = createVectorDB();
    
    // Insert vectors
    int num_vectors = 10000000;  // 10 million vectors for this example
    for (int i = 0; i < num_vectors; i++) {
        Vector v;
        generateRandomVector(&v);
        insertVector(db, &v);
        if (i % 100000 == 0) {
            printf("Inserted %d vectors\n", i);
        }
    }
    
    // Perform sample queries
    int num_queries = 100;
    double total_time = 0;
    
    for (int i = 0; i < num_queries; i++) {
        Vector query;
        generateRandomVector(&query);
        
        clock_t start = clock();
        Vector* nearest = findNearest(db, &query);
        clock_t end = clock();
        
        double query_time = ((double) (end - start)) / CLOCKS_PER_SEC;
        total_time += query_time;
        
        printf("Query %d: Nearest vector found. ID: %d, Time: %f seconds\n", i + 1, nearest->id, query_time);
    }
    
    printf("Average query time: %f seconds\n", total_time / num_queries);
    
    freeVectorDB(db);
    return 0;
}
